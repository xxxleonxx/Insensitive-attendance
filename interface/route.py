

ROUTE = \
    {
        1000: '/non_inductive_attendance/remote/local/tag',            # 配置主机唯一标识

        1001: '/non_inductive_attendance/remote/config',               # 查询、配置远程参数

        1002: '/non_inductive_attendance/function/status',             # 获取功能状态

        2001: '/non_inductive_attendance/workers',                     # 获取劳务人员列表
        2002: '/non_inductive_attendance/devices',                     # 获取设备信息列表

        2003: '/non_inductive_attendance/get/workers/idcard',          # 通过身份证号获取人员信息

        3001: '/non_inductive_attendance/workers_info/add',            # 劳务人员信息添加接口
        3002: '/non_inductive_attendance/workers_info/delete',         # 劳务人员信息删除接口

        4001: '/non_inductive_attendance/access/add',                  # 添加闸机设备信息
        4002: '/non_inductive_attendance/access/delete',               # 删除闸机设备信息

        5001: '/non_inductive_attendance/camera/add',                  # 添加相机信息
        5002: '/non_inductive_attendance/camera/delete',               # 删除相机信息

        6001: '/non_inductive_attendance/control/identify/open',       # 启动人脸识别功能
        6002: '/non_inductive_attendance/control/identify/close',      # 关闭人脸识别功能

        7001: '/non_inductive_attendance/control/push/open',           # 启动数据推送功能
        7002: '/non_inductive_attendance/control/push/close',          # 关闭数据推送功能

        8001: '/non_inductive_attendance/control/sync/open',           # 启动同步劳务人员功能
        8002: '/non_inductive_attendance/control/sync/close',          # 关闭同步劳务人员功能

        9001: '/non_inductive_attendance/control/switch/open',         # 启动闸机控制功能
        9002: '/non_inductive_attendance/control/switch/close',        # 关闭闸机控制功能

        10001: '/non_inductive_attendance/get/identify/threshold',     # 获取人脸识别阈值
        10002: '/non_inductive_attendance/set/identify/threshold',     # 设置人脸识别阈值
    }
