from typing import Iterable, List, Tuple

from ultralytics import YOLOv10
import supervision as sv
import numpy as np
import re


def initiate_polygon_zones(polygons: List[np.ndarray],
                           triggering_anchors: Iterable[sv.Position] = sv.Position.CENTER) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(polygon=polygon, triggering_anchors=triggering_anchors)
        for polygon in polygons
    ]


class CountObject:

    def __init__(self, input_video_path, classes_to_count, polygon_zone) -> None:
        self.model = YOLOv10('yolov10m.pt')
        self.colors = sv.ColorPalette.DEFAULT

        self.input_video_path = input_video_path
        self.classes_to_count = classes_to_count

        pattern = r'\d+'
        values = re.findall(pattern, polygon_zone)
        values = list(map(int, values))
        array = np.array(values).reshape(4, 2)

        self.polygons = [np.array(array)]

        self.video_info = sv.VideoInfo.from_video_path(input_video_path)
        self.zones = initiate_polygon_zones(polygons=self.polygons,
                                            triggering_anchors=[sv.Position.CENTER])

        self.zone_annotators = [
            sv.PolygonZoneAnnotator(zone=zone,
                                    color=self.colors.by_idx(index),
                                    thickness=6,
                                    text_thickness=8,
                                    text_scale=4)
            for index, zone in enumerate(self.zones)
        ]

        self.box_annotators = [
            sv.BoundingBoxAnnotator(color=self.colors.by_idx(index), thickness=4)
            for index in range(len(self.polygons))
        ]

    def process_frame(self, frame: np.ndarray, _) -> np.ndarray:
        results = self.model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[(detections.class_id == self.classes_to_count) & (detections.confidence > 0.5)]

        for zone, zone_annotator, box_annotator in zip(self.zones, self.zone_annotators, self.box_annotators):
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            frame = box_annotator.annotate(scene=frame, detections=detections_filtered)
            frame = zone_annotator.annotate(scene=frame)

        return frame

    def process_video(self):
        sv.process_video(source_path=self.input_video_path, target_path='output.mp4', callback=self.process_frame)


def process_video_and_count(video_path, classes_to_count=None, polygon_zone=None):
    """
    Process the video to count objects, draw bounding boxes around detected objects,
    and save an annotated video along with a JSON file containing the counts.
    """

    if classes_to_count is None:
        classes_to_count = [i for i in range(1, 81)]

    co = CountObject(video_path, classes_to_count, polygon_zone)
    co.process_video()

    return 'output.mp4'
