from detectron2.data import MetadataCatalog

COCO_PERSON_KEYPOINT_NAMES = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)
# fmt: on

# Pairs of keypoints that should be exchanged under horizontal flipping
COCO_PERSON_KEYPOINT_FLIP_MAP = (
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
)

# rules for pairs of keypoints to draw a line between, and the line color to use.
KEYPOINT_CONNECTION_RULES = [
    # face
    ("left_ear", "left_eye", (102, 204, 255)),
    ("right_ear", "right_eye", (51, 153, 255)),
    ("left_eye", "nose", (102, 0, 204)),
    ("nose", "right_eye", (51, 102, 255)),
    # upper-body
    ("left_shoulder", "right_shoulder", (255, 128, 0)),
    ("left_shoulder", "left_elbow", (153, 255, 204)),
    ("right_shoulder", "right_elbow", (128, 229, 255)),
    ("left_elbow", "left_wrist", (153, 255, 153)),
    ("right_elbow", "right_wrist", (102, 255, 224)),
    # lower-body
    ("left_hip", "right_hip", (255, 102, 0)),
    ("left_hip", "left_knee", (255, 255, 77)),
    ("right_hip", "right_knee", (153, 255, 204)),
    ("left_knee", "left_ankle", (191, 255, 128)),
    ("right_knee", "right_ankle", (255, 195, 77)),
]


MetadataCatalog.get("densepose_coco_2014_train").keypoint_names = COCO_PERSON_KEYPOINT_NAMES
MetadataCatalog.get("densepose_coco_2014_train").keypoint_flip_map = COCO_PERSON_KEYPOINT_FLIP_MAP
MetadataCatalog.get("densepose_coco_2014_train").keypoint_connection_rules = KEYPOINT_CONNECTION_RULES

MetadataCatalog.get("densepose_coco_2014_minival").keypoint_names = COCO_PERSON_KEYPOINT_NAMES
MetadataCatalog.get("densepose_coco_2014_minival").keypoint_flip_map = COCO_PERSON_KEYPOINT_FLIP_MAP
MetadataCatalog.get("densepose_coco_2014_minival").keypoint_connection_rules = KEYPOINT_CONNECTION_RULES

MetadataCatalog.get("densepose_coco_2014_valminusminival").keypoint_names = COCO_PERSON_KEYPOINT_NAMES
MetadataCatalog.get("densepose_coco_2014_valminusminival").keypoint_flip_map = COCO_PERSON_KEYPOINT_FLIP_MAP
MetadataCatalog.get("densepose_coco_2014_valminusminival").keypoint_connection_rules = KEYPOINT_CONNECTION_RULES

MetadataCatalog.get("densepose_coco_2014_minival_100").keypoint_names = COCO_PERSON_KEYPOINT_NAMES
MetadataCatalog.get("densepose_coco_2014_minival_100").keypoint_flip_map = COCO_PERSON_KEYPOINT_FLIP_MAP
MetadataCatalog.get("densepose_coco_2014_minival_100").keypoint_connection_rules = KEYPOINT_CONNECTION_RULES