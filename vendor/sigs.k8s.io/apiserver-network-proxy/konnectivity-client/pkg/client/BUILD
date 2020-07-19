load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "client.go",
        "conn.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/apiserver-network-proxy/konnectivity-client/pkg/client",
    importpath = "sigs.k8s.io/apiserver-network-proxy/konnectivity-client/pkg/client",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/google.golang.org/grpc:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
        "//vendor/sigs.k8s.io/apiserver-network-proxy/konnectivity-client/proto/client:go_default_library",
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
