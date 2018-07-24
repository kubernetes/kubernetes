load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "descriptor.pb.go",
        "doc.go",
        "metrics.pb.go",
        "mount.pb.go",
        "platform.pb.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/containerd/containerd/api/types",
    importpath = "github.com/containerd/containerd/api/types",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/gogo/protobuf/proto:go_default_library",
        "//vendor/github.com/gogo/protobuf/types:go_default_library",
        "//vendor/github.com/opencontainers/go-digest:go_default_library",
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
        "//vendor/github.com/containerd/containerd/api/types/task:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
