load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["autoscaling_utils.go"],
    importpath = "k8s.io/kubernetes/test/e2e/framework/autoscaling",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/autoscaling/v1:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/scale:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//test/e2e/framework/kubectl:go_default_library",
        "//test/e2e/framework/rc:go_default_library",
        "//test/e2e/framework/resource:go_default_library",
        "//test/e2e/framework/service:go_default_library",
        "//test/utils:go_default_library",
        "//test/utils/image:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
