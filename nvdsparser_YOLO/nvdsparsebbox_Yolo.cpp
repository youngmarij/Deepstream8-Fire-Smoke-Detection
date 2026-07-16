#include "nvdsinfer_custom_impl.h"
#include "utils.h"

extern "C" bool
NvDsInferParseYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

static NvDsInferParseObjectInfo
convertBBox(
    const float& bx1,
    const float& by1,
    const float& bx2,
    const float& by2,
    const uint& netW,
    const uint& netH)
{
    NvDsInferParseObjectInfo b;

    float x1 = clamp(bx1, 0.0f, (float)netW);
    float y1 = clamp(by1, 0.0f, (float)netH);
    float x2 = clamp(bx2, 0.0f, (float)netW);
    float y2 = clamp(by2, 0.0f, (float)netH);

    b.left = x1;
    b.top = y1;
    b.width = clamp(x2 - x1, 0.0f, (float)netW);
    b.height = clamp(y2 - y1, 0.0f, (float)netH);

    return b;
}

static void
addBBoxProposal(
    float bx1,
    float by1,
    float bx2,
    float by2,
    const uint& netW,
    const uint& netH,
    int classId,
    float confidence,
    std::vector<NvDsInferParseObjectInfo>& objects)
{
    NvDsInferParseObjectInfo obj =
        convertBBox(bx1, by1, bx2, by2, netW, netH);

    if (obj.width < 1 || obj.height < 1)
        return;

    obj.classId = classId;
    obj.detectionConfidence = confidence;

    objects.push_back(obj);
}

static std::vector<NvDsInferParseObjectInfo>
decodeTensorYolo(
    const float* output,
    const uint& numBoxes,
    const uint& netW,
    const uint& netH,
    const std::vector<float>& thresholds)
{
    std::vector<NvDsInferParseObjectInfo> objects;

    for (uint i = 0; i < numBoxes; i++)
    {
        float xc = output[0 * numBoxes + i];
        float yc = output[1 * numBoxes + i];
        float w  = output[2 * numBoxes + i];
        float h  = output[3 * numBoxes + i];

        float fire  = output[4 * numBoxes + i];
        float other = output[5 * numBoxes + i];
        float smoke = output[6 * numBoxes + i];

        int cls = 0;
        float score = fire;

        if (other > score)
        {
            score = other;
            cls = 1;
        }

        if (smoke > score)
        {
            score = smoke;
            cls = 2;
        }

        if (score < thresholds[cls])
            continue;

        float x1 = xc - w * 0.5f;
        float y1 = yc - h * 0.5f;
        float x2 = xc + w * 0.5f;
        float y2 = yc + h * 0.5f;

        addBBoxProposal(
            x1,
            y1,
            x2,
            y2,
            netW,
            netH,
            cls,
            score,
            objects);
    }

    return objects;
}

static bool
NvDsInferParseCustomYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayersInfo.empty())
    {
        std::cerr << "ERROR: Output layer not found." << std::endl;
        return false;
    }

    const NvDsInferLayerInfo& output = outputLayersInfo[0];

    const uint numBoxes = output.inferDims.d[1];

    objectList = decodeTensorYolo(
        static_cast<const float*>(output.buffer),
        numBoxes,
        networkInfo.width,
        networkInfo.height,
        detectionParams.perClassPreclusterThreshold);

    return true;
}

extern "C" bool
NvDsInferParseYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    return NvDsInferParseCustomYolo(
        outputLayersInfo,
        networkInfo,
        detectionParams,
        objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo);
