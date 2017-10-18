package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "scheduler_perf_types.go",
        "util.go",
    ],
    importpath = "k8s.io/kubernetes/test/integration/scheduler_perf",
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//plugin/pkg/scheduler:go_default_library",
        "//plugin/pkg/scheduler/algorithmprovider:go_default_library",
        "//plugin/pkg/scheduler/factory:go_default_library",
        "//test/integration/framework:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/client-go/informers:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/core/v1:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
        "//vendor/k8s.io/client-go/tools/record:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    size = "large",
    srcs = [
        "main_test.go",
        "scheduler_bench_test.go",
        "scheduler_test.go",
    ],
    importpath = "k8s.io/kubernetes/test/integration/scheduler_perf",
    library = ":go_default_library",
    tags = ["integration"],
    deps = [
        "//plugin/pkg/scheduler:go_default_library",
        "//test/integration/framework:go_default_library",
        "//test/utils:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
