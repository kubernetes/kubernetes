package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["util.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//plugin/pkg/scheduler:go_default_library",
        "//plugin/pkg/scheduler/algorithmprovider:go_default_library",
        "//plugin/pkg/scheduler/factory:go_default_library",
        "//test/integration/framework:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "scheduler_bench_test.go",
        "scheduler_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//plugin/pkg/scheduler:go_default_library",
        "//test/integration/framework:go_default_library",
        "//test/utils:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
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
