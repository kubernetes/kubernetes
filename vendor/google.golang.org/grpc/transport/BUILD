load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "control.go",
        "go16.go",
        "go17.go",
        "handler_server.go",
        "http2_client.go",
        "http2_server.go",
        "http_util.go",
        "pre_go16.go",
        "transport.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/golang.org/x/net/http2:go_default_library",
        "//vendor/golang.org/x/net/http2/hpack:go_default_library",
        "//vendor/golang.org/x/net/trace:go_default_library",
        "//vendor/google.golang.org/grpc/codes:go_default_library",
        "//vendor/google.golang.org/grpc/credentials:go_default_library",
        "//vendor/google.golang.org/grpc/grpclog:go_default_library",
        "//vendor/google.golang.org/grpc/metadata:go_default_library",
        "//vendor/google.golang.org/grpc/peer:go_default_library",
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
