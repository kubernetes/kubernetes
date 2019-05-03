load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "any.go",
        "any.pb.go",
        "api.pb.go",
        "doc.go",
        "duration.go",
        "duration.pb.go",
        "duration_gogo.go",
        "empty.pb.go",
        "field_mask.pb.go",
        "protosize.go",
        "source_context.pb.go",
        "struct.pb.go",
        "timestamp.go",
        "timestamp.pb.go",
        "timestamp_gogo.go",
        "type.pb.go",
        "wrappers.pb.go",
        "wrappers_gogo.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/gogo/protobuf/types",
    importpath = "github.com/gogo/protobuf/types",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/gogo/protobuf/proto:go_default_library",
        "//vendor/github.com/gogo/protobuf/sortkeys:go_default_library",
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
