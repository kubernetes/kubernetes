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
        "config.go",
        "configuration_map.go",
        "doc.go",
        "feature_gate.go",
        "namedcertkey_flag.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/util/wait:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/spf13/pflag",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "config_test.go",
        "feature_gate_test.go",
        "namedcertkey_flag_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = ["//vendor:github.com/spf13/pflag"],
)
