load("@io_bazel_rules_go//go:def.bzl", "go_library")

filegroup(
    name = "go_default_library_protos",
    srcs = ["api.proto"],
    visibility = ["//visibility:public"],
)

go_library(
    name = "go_default_library",
    srcs = ["api.pb.go"],
    importpath = "github.com/coreos/rkt/api/v1alpha",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/google.golang.org/grpc:go_default_library",
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
