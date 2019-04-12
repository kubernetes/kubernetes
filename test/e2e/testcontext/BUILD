load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["test_context.go"],
    importpath = "k8s.io/kubernetes/test/e2e/testcontext",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubelet/apis/config:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/client-go/tools/clientcmd:go_default_library",
        "//staging/src/k8s.io/client-go/tools/clientcmd/api:go_default_library",
        "//staging/src/k8s.io/component-base/cli/flag:go_default_library",
        "//vendor/github.com/onsi/ginkgo/config:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
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
    visibility = ["//visibility:public"],
)
