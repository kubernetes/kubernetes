load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "authorization_manager.go",
        "datacenter.go",
        "doc.go",
        "event_manager.go",
        "host_config_info.go",
        "host_firewall_system.go",
        "host_hardware_info.go",
        "host_storage_device_info.go",
        "host_system.go",
        "performance_manager.go",
        "performance_manager_data.go",
        "resource_pool.go",
        "root_folder.go",
        "service_content.go",
        "setting.go",
        "task_manager.go",
        "virtual_device.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/vmware/govmomi/simulator/esx",
    importpath = "github.com/vmware/govmomi/simulator/esx",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/vmware/govmomi/vim25/mo:go_default_library",
        "//vendor/github.com/vmware/govmomi/vim25/types:go_default_library",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
