package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

filegroup(
    name = "config",
    srcs = glob([
        "**/*.yaml",
        "**/*.yml",
        "**/*.json",
    ]) + [
        "pod",
    ],
)

filegroup(
    name = "sources",
    srcs = glob([
        "**/*",
    ]),
)

go_library(
    name = "go_default_library",
    srcs = ["doc.go"],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_xtest",
    srcs = ["examples_test.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/validation:go_default_library",
        "//pkg/apis/apps:go_default_library",
        "//pkg/apis/apps/validation:go_default_library",
        "//pkg/apis/batch:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/extensions/validation:go_default_library",
        "//pkg/capabilities:go_default_library",
        "//pkg/registry/batch/job:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/validation/field:go_default_library",
        "//pkg/util/yaml:go_default_library",
        "//plugin/pkg/scheduler/api:go_default_library",
        "//plugin/pkg/scheduler/api/latest:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
