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
        "doc.go",
        "labels.go",
        "selector.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/selection:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/validation:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "labels_test.go",
        "selector_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/selection:go_default_library",
        "//pkg/util/sets:go_default_library",
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
