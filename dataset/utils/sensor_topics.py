from utils import Sensors
from environment_model import ReadingTypes

# Camera and Radar object topics
asw_topics_cam_radar = [
    (
        ["Logdata", "Vehicle", "GenericInterface", "CAN", "81_Vehicle_Input_PrioA_02"],
        ReadingTypes.Vehicle_Input_PrioA_02,
        Sensors.VEHICLE_INTERNAL,
    ),
    (
        ["Logdata", "Camera_FC0", "Objects", "ASW_VidObjData_t"],
        ReadingTypes.ASW_VidObjData_t,
        Sensors.CAMERA_FC0,
    ),
    (
        ["Logdata", "Radar_FL0", "Objects", "ASW_RdrObjDataAppl_t"],
        ReadingTypes.ASW_RdrObjDataAppl_t,
        Sensors.RADAR_FL0,
    ),
    (
        ["Logdata", "Radar_FR0", "Objects", "ASW_RdrObjDataAppl_t"],
        ReadingTypes.ASW_RdrObjDataAppl_t,
        Sensors.RADAR_FR0,
    ),
    (
        ["Logdata", "Radar_RL0", "Objects", "ASW_RdrObjDataAppl_t"],
        ReadingTypes.ASW_RdrObjDataAppl_t,
        Sensors.RADAR_RL0,
    ),
    (
        ["Logdata", "Radar_RR0", "Objects", "ASW_RdrObjDataAppl_t"],
        ReadingTypes.ASW_RdrObjDataAppl_t,
        Sensors.RADAR_RR0,
    ),
]

# Object topics of Lidar ground truth
asw_topics_lidar_gt = [
    (
        ["Logdata", "Vehicle", "GenericInterface", "CAN", "81_Vehicle_Input_PrioA_02"],
        ReadingTypes.Vehicle_Input_PrioA_02,
        Sensors.VEHICLE_INTERNAL,
    ),
    (
        ["Logdata", "Lidar_GT", "Objects", "ASW_LdrObjData_t"],
        ReadingTypes.ASW_LdrObjData_t,
        Sensors.LIDAR_GT,
    ),
]

"81_Vehicle_Input_PrioA_02 -> HstVehTurnSigIdcrSts"
raw_topics = [
    (["lidar_data__LUX_00", "LidarObjects"], ReadingTypes.LIDAR_OBJECT, Sensors.LIDAR),
    (
        ["lidar_data__LUX_00", "Lidar_GPS_IMU"],
        ReadingTypes.LIDAR_GPS_IMU,
        Sensors.LIDAR,
    ),
    (["lanes__EPM_01_LS", "Lanes"], ReadingTypes.LANE, Sensors.CAMERA),
    (["obstacles__EPM_01_OBS", "Obstacles"], ReadingTypes.OBSTACLE, Sensors.CAMERA),
    (["ac1000_mrr__AC1000_MRR_01", "Tracks"], ReadingTypes.TRACK, Sensors.RADAR),
]


# # All the basic topics
# asw_topics = [

#     (['Logdata', 'Vehicle', 'GenericInterface', 'CAN', '81_Vehicle_Input_PrioA_02'],
#      ReadingTypes.Vehicle_Input_PrioA_02, Sensors.VEHICLE_INTERNAL),

#     (['Logdata', 'Camera_FC0', 'Lines', 'ASW_VidLineData_t'], ReadingTypes.ASW_VidLineData_t,
#      Sensors.CAMERA_FC0),
#     (['Logdata', 'Camera_FC0', 'Objects', 'ASW_VidObjData_t'], ReadingTypes.ASW_VidObjData_t,
#      Sensors.CAMERA_FC0),
#     (['Logdata', 'Camera_FC0', 'TrafficSigns', 'ASW_VidTrffcSignData_t'], ReadingTypes.ASW_VidTrffcSignData_t,
#      Sensors.CAMERA_FC0),
#     (['Logdata', 'Camera_FC0', 'TrafficLights', 'ASW_VidTrafficLightData_t'], ReadingTypes.ASW_VidTrffcLightData_t,
#      Sensors.CAMERA_FC0),
#     (['Logdata', 'Lidar_GT', 'Objects', 'ASW_LdrObjData_t'], ReadingTypes.ASW_LdrObjData_t,
#      Sensors.LIDAR_GT),
#     (['Logdata', 'Lidar_SR0', 'Objects', 'ASW_LdrObjData_t'], ReadingTypes.ASW_LdrObjData_t,
#      Sensors.LIDAR_SR0),

#     (['Logdata', 'Radar_FC0', 'Objects', 'ASW_RdrObjData_t'], ReadingTypes.ASW_RdrObjData_t,
#      Sensors.RADAR_FC0),
#     (['Logdata', 'Radar_FC0', 'Freespace', 'ASW_RdrFrespcData_t'], ReadingTypes.ASW_RdrFrespcData_t,
#      Sensors.RADAR_FC0),
#     (['Logdata', 'Radar_FC0', 'RoadEdge', 'ASW_RdrRoadEdgeData_t'], ReadingTypes.ASW_RdrRoadEdgeData_t,
#      Sensors.RADAR_FC0),

#     (['Logdata', 'Radar_FL0', 'Objects', 'ASW_RdrObjDataAppl_t'], ReadingTypes.ASW_RdrObjDataAppl_t,
#      Sensors.RADAR_FL0),

#     (['Logdata', 'Radar_FR0', 'Objects', 'ASW_RdrObjDataAppl_t'], ReadingTypes.ASW_RdrObjDataAppl_t,
#      Sensors.RADAR_FR0),

#     (['Logdata', 'Radar_RL0', 'Objects', 'ASW_RdrObjDataAppl_t'], ReadingTypes.ASW_RdrObjDataAppl_t,
#      Sensors.RADAR_RL0),

#     (['Logdata', 'Radar_RR0', 'Objects', 'ASW_RdrObjDataAppl_t'], ReadingTypes.ASW_RdrObjDataAppl_t,
#      Sensors.RADAR_RR0),

#     (['Logdata', 'Vehicle', 'GenericInterface', 'CAN', '80_Vehicle_Input_PrioA_01'],
#      ReadingTypes.Vehicle_Input_PrioA_01, Sensors.VEHICLE_INTERNAL),

#     (['Logdata', 'Here_API', 'Environment', 'ASW_HereAPIData_t'],
#      ReadingTypes.ASW_HereAPIData_t, Sensors.HERE_API),

# ]
