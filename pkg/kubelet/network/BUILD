package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "network.go",
        "plugins.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/componentconfig:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/sysctl:go_default_library",
        "//pkg/util/validation:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
