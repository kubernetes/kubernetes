load("@io_bazel_rules_go//go:def.bzl", "go_library")

filegroup(
    name = "go_default_library_protos",
    srcs = ["multilog.proto"],
    visibility = ["//visibility:public"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "gen.go",
        "multilog.pb.go",
    ],
    importpath = "github.com/google/certificate-transparency-go/client/configpb",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
        "//vendor/github.com/golang/protobuf/ptypes/timestamp:go_default_library",
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
