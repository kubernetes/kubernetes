package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "extender_test.go",
        "generic_scheduler_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/apps/v1beta1:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//plugin/pkg/scheduler/algorithm:go_default_library",
        "//plugin/pkg/scheduler/algorithm/predicates:go_default_library",
        "//plugin/pkg/scheduler/algorithm/priorities:go_default_library",
        "//plugin/pkg/scheduler/algorithm/priorities/util:go_default_library",
        "//plugin/pkg/scheduler/api:go_default_library",
        "//plugin/pkg/scheduler/schedulercache:go_default_library",
        "//plugin/pkg/scheduler/testing:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "equivalence_cache.go",
        "extender.go",
        "generic_scheduler.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/util/hash:go_default_library",
        "//plugin/pkg/scheduler/algorithm:go_default_library",
        "//plugin/pkg/scheduler/algorithm/predicates:go_default_library",
        "//plugin/pkg/scheduler/api:go_default_library",
        "//plugin/pkg/scheduler/schedulercache:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/golang/groupcache/lru",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apiserver/pkg/util/trace",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/util/workqueue",
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
