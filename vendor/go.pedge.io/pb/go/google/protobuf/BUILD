load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "any.pb.go",
        "api.pb.go",
        "duration.pb.go",
        "empty.pb.go",
        "field_mask.pb.go",
        "protobuf.gen.go",
        "source_context.pb.go",
        "struct.pb.go",
        "timestamp.pb.go",
        "type.pb.go",
        "wrappers.pb.go",
    ],
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/golang/protobuf/proto:go_default_library"],
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
