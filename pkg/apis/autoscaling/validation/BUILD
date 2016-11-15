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
    srcs = ["validation.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/validation:go_default_library",
        "//pkg/api/validation/path:go_default_library",
        "//pkg/apis/autoscaling:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/controller/podautoscaler:go_default_library",
        "//pkg/util/validation/field:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["validation_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apis/autoscaling:go_default_library",
        "//pkg/controller/podautoscaler:go_default_library",
    ],
)
