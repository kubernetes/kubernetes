load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "framework.go",
        "imports.go",
        "nodes.go",
    ],
    importpath = "k8s.io/kubernetes/test/e2e/cloud",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//test/e2e/cloud/gcp:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//test/e2e/framework/node:go_default_library",
        "//test/e2e/framework/skipper:go_default_library",
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
        "//test/e2e/cloud/gcp:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
