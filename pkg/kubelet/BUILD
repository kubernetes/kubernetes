package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "active_deadline.go",
        "disk_manager.go",
        "doc.go",
        "kubelet.go",
        "kubelet_cadvisor.go",
        "kubelet_getters.go",
        "kubelet_network.go",
        "kubelet_node_status.go",
        "kubelet_pods.go",
        "kubelet_resources.go",
        "kubelet_volumes.go",
        "networks.go",
        "oom_watcher.go",
        "pod_container_deletor.go",
        "pod_workers.go",
        "reason_cache.go",
        "runonce.go",
        "runtime.go",
        "util.go",
        "volume_host.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "active_deadline_test.go",
        "disk_manager_test.go",
        "kubelet_cadvisor_test.go",
        "kubelet_getters_test.go",
        "kubelet_network_test.go",
        "kubelet_node_status_test.go",
        "kubelet_pods_test.go",
        "kubelet_resources_test.go",
        "kubelet_test.go",
        "kubelet_volumes_test.go",
        "oom_watcher_test.go",
        "pod_container_deletor_test.go",
        "pod_workers_test.go",
        "reason_cache_test.go",
        "runonce_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
