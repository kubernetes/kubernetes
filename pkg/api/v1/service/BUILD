package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "annotations.go",
        "util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/util/net/sets:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apis/pkg/api/v1",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["util_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/util/net/sets:go_default_library",
        "//vendor:k8s.io/apis/pkg/api/v1",
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
)
