package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "events.go",
        "imports.go",
    ],
    importpath = "k8s.io/kubernetes/test/e2e/instrumentation",
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/events/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/typed/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/typed/events/v1:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//test/e2e/instrumentation/common:go_default_library",
        "//test/e2e/instrumentation/logging:go_default_library",
        "//test/e2e/instrumentation/monitoring:go_default_library",
        "//vendor/github.com/onsi/ginkgo:go_default_library",
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
        "//test/e2e/instrumentation/common:all-srcs",
        "//test/e2e/instrumentation/logging:all-srcs",
        "//test/e2e/instrumentation/monitoring:all-srcs",
    ],
    tags = ["automanaged"],
)
