load("@io_bazel_rules_go//go:def.bzl", "go_library")

filegroup(
    name = "go_default_library_protos",
    srcs = [
        "descriptor.proto",
        "metrics.proto",
        "mount.proto",
        "platform.proto",
    ],
    visibility = ["//visibility:public"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "descriptor.pb.go",
        "doc.go",
        "metrics.pb.go",
        "mount.pb.go",
        "platform.pb.go",
    ],
    importpath = "github.com/containerd/containerd/api/types",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/gogo/protobuf/gogoproto:go_default_library",
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
