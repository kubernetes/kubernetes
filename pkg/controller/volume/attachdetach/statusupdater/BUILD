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
        "fake_node_status_updater.go",
        "node_status_updater.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/release_1_5:go_default_library",
        "//pkg/controller/volume/attachdetach/cache:go_default_library",
        "//pkg/conversion:go_default_library",
        "//pkg/util/strategicpatch:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
