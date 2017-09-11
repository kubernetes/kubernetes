load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "topology.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubelet/cm/cpuset:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = ["topology_test.go"],
    library = ":go_default_library",
    deps = ["//vendor/github.com/google/cadvisor/info/v1:go_default_library"],
)
