load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "isolate_linux.go",
        "isolate_stub.go",
        "netutil.go",
        "routes.go",
        "routes_linux.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/pkg/netutil",
    importpath = "go.etcd.io/etcd/pkg/netutil",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.etcd.io/etcd/pkg/types:go_default_library",
        "//vendor/go.uber.org/zap:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/go.etcd.io/etcd/pkg/cpuutil:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/go.etcd.io/etcd/pkg/cpuutil:go_default_library",
        ],
        "//conditions:default": [],
    }),
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
