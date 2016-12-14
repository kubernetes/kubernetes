package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "fake_node_status_updater.go",
        "node_status_updater.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/controller/volume/attachdetach/cache:go_default_library",
        "//pkg/util/strategicpatch:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
