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
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/selection:go_default_library",
        "//pkg/util/sets:go_default_library",
    ],
)
