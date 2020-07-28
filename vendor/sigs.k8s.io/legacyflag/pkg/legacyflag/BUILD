load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bool.go",
        "bool_slice.go",
        "flag.go",
        "float32.go",
        "float64.go",
        "global.go",
        "int.go",
        "int16.go",
        "int32.go",
        "int64.go",
        "int8.go",
        "int_slice.go",
        "map.go",
        "map_string_bool.go",
        "map_string_string.go",
        "net_ip.go",
        "net_ipnet.go",
        "string.go",
        "string_slice.go",
        "time_duration.go",
        "uint.go",
        "uint16.go",
        "uint32.go",
        "uint64.go",
        "uint8.go",
        "uint_slice.go",
        "var.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/legacyflag/pkg/legacyflag",
    importpath = "sigs.k8s.io/legacyflag/pkg/legacyflag",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/spf13/pflag:go_default_library"],
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
