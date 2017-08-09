package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "scheduler_interface.go",
        "types.go",
        "well_known_labels.go",
    ],
    deps = [
        "//plugin/pkg/scheduler/api:go_default_library",
        "//plugin/pkg/scheduler/schedulercache:go_default_library",
        "//vendor/k8s.io/api/apps/v1beta1:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/api/extensions/v1beta1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "scheduler_interface_test.go",
        "types_test.go",
    ],
    library = ":go_default_library",
    deps = [
        "//plugin/pkg/scheduler/schedulercache:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//plugin/pkg/scheduler/algorithm/predicates:all-srcs",
        "//plugin/pkg/scheduler/algorithm/priorities:all-srcs",
    ],
    tags = ["automanaged"],
)
