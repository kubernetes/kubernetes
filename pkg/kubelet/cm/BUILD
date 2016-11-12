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
        "cgroup_manager_linux.go",
        "container_manager.go",
        "container_manager_linux.go",
        "container_manager_stub.go",
        "helpers_linux.go",
        "pod_container_manager_linux.go",
        "pod_container_manager_stub.go",
        "types.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "cgroup_manager_linux_test.go",
        "container_manager_linux_test.go",
        "helpers_linux_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
