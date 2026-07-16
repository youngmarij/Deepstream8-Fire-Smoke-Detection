import pyds
import datetime
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
from parser import parse_args
args = parse_args()

def pgie_src_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return
    
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            l_obj = frame_meta.obj_meta_list
            print(
                f"Frame {frame_meta.frame_num}"
                f"num_obj_meta = {frame_meta.num_obj_meta}"
            )
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        print(
            "Frame Number=",
            frame_number
        )
        if args.rtsp_ts:
            ts = frame_meta.ntp_timestamp/1000000000 # Retrieve timestamp, put decimal in proper position for Unix format
            print("RTSP Timestamp:",datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')) # Convert timestamp to UTC

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK