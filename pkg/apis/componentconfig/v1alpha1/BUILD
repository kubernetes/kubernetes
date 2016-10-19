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
        "defaults.go",
        "doc.go",
        "register.go",
        "types.go",
        "zz_generated.conversion.go",
        "zz_generated.deepcopy.go",
        "zz_generated.defaults.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/apis/componentconfig:go_default_library",
        "//pkg/conversion:go_default_library",
        "//pkg/kubelet/qos:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//pkg/master/ports:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/config:go_default_library",
    ],
)
