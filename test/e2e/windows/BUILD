package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "density.go",
        "framework.go",
        "gmsa.go",
        "hybrid_network.go",
        "memory_limits.go",
        "networking.go",
        "service.go",
        "volumes.go",
    ],
    importpath = "k8s.io/kubernetes/test/e2e/windows",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubelet/apis/config:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/scheme:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/kubelet/config/v1beta1:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//test/e2e/framework/kubelet:go_default_library",
        "//test/e2e/framework/log:go_default_library",
        "//test/e2e/framework/metrics:go_default_library",
        "//test/e2e/framework/node:go_default_library",
        "//test/e2e/framework/pod:go_default_library",
        "//test/e2e/framework/service:go_default_library",
        "//test/utils/image:go_default_library",
        "//vendor/github.com/onsi/ginkgo:go_default_library",
        "//vendor/github.com/onsi/gomega:go_default_library",
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
