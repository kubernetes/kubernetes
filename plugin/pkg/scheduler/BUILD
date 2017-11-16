package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["scheduler_test.go"],
    importpath = "k8s.io/kubernetes/plugin/pkg/scheduler",
    library = ":go_default_library",
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//plugin/pkg/scheduler/algorithm:go_default_library",
        "//plugin/pkg/scheduler/algorithm/predicates:go_default_library",
        "//plugin/pkg/scheduler/core:go_default_library",
        "//plugin/pkg/scheduler/schedulercache:go_default_library",
        "//plugin/pkg/scheduler/testing:go_default_library",
        "//plugin/pkg/scheduler/util:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/k8s.io/client-go/tools/record:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "scheduler.go",
        "testutil.go",
    ],
    importpath = "k8s.io/kubernetes/plugin/pkg/scheduler",
    deps = [
        "//pkg/features:go_default_library",
        "//plugin/pkg/scheduler/algorithm:go_default_library",
        "//plugin/pkg/scheduler/api:go_default_library",
        "//plugin/pkg/scheduler/core:go_default_library",
        "//plugin/pkg/scheduler/metrics:go_default_library",
        "//plugin/pkg/scheduler/schedulercache:go_default_library",
        "//plugin/pkg/scheduler/util:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/client-go/listers/core/v1:go_default_library",
        "//vendor/k8s.io/client-go/tools/record:go_default_library",
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
        "//plugin/pkg/scheduler/algorithm:all-srcs",
        "//plugin/pkg/scheduler/algorithmprovider:all-srcs",
        "//plugin/pkg/scheduler/api:all-srcs",
        "//plugin/pkg/scheduler/core:all-srcs",
        "//plugin/pkg/scheduler/factory:all-srcs",
        "//plugin/pkg/scheduler/metrics:all-srcs",
        "//plugin/pkg/scheduler/schedulercache:all-srcs",
        "//plugin/pkg/scheduler/testing:all-srcs",
        "//plugin/pkg/scheduler/util:all-srcs",
    ],
    tags = ["automanaged"],
)
