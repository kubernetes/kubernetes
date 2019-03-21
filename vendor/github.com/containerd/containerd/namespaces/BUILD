load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "context.go",
        "grpc.go",
        "store.go",
        "validate.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/containerd/containerd/namespaces",
    importpath = "github.com/containerd/containerd/namespaces",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/containerd/containerd/errdefs:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/google.golang.org/grpc/metadata:go_default_library",
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
