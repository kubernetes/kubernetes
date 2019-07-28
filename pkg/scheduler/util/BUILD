package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "error_channel_test.go",
        "heap_test.go",
        "utils_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/apis/scheduling:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "clock.go",
        "error_channel.go",
        "heap.go",
        "utils.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/scheduler/util",
    deps = [
        "//pkg/apis/scheduling:go_default_library",
        "//pkg/scheduler/api:go_default_library",
        "//pkg/scheduler/metrics:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
