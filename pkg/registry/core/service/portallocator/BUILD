package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "allocator.go",
        "operation.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/registry/core/service/portallocator",
    deps = [
        "//pkg/apis/core:go_default_library",
        "//pkg/registry/core/service/allocator:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/net:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "allocator_test.go",
        "operation_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/apis/core:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/net:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
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
        "//pkg/registry/core/service/portallocator/controller:all-srcs",
    ],
    tags = ["automanaged"],
)
