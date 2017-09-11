package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["allocator.go"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/registry/core/service/allocator:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["allocator_test.go"],
    library = ":go_default_library",
    deps = [
        "//pkg/api:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
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
        "//pkg/registry/core/service/ipallocator/controller:all-srcs",
        "//pkg/registry/core/service/ipallocator/storage:all-srcs",
    ],
    tags = ["automanaged"],
)
