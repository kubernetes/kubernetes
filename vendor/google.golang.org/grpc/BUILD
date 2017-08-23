load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "backoff.go",
        "balancer.go",
        "call.go",
        "clientconn.go",
        "doc.go",
        "interceptor.go",
        "rpc_util.go",
        "server.go",
        "stream.go",
        "trace.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/golang.org/x/net/http2:go_default_library",
        "//vendor/golang.org/x/net/trace:go_default_library",
        "//vendor/google.golang.org/grpc/codes:go_default_library",
        "//vendor/google.golang.org/grpc/credentials:go_default_library",
        "//vendor/google.golang.org/grpc/grpclog:go_default_library",
        "//vendor/google.golang.org/grpc/internal:go_default_library",
        "//vendor/google.golang.org/grpc/metadata:go_default_library",
        "//vendor/google.golang.org/grpc/naming:go_default_library",
        "//vendor/google.golang.org/grpc/transport:go_default_library",
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
        "//vendor/google.golang.org/grpc/codes:all-srcs",
        "//vendor/google.golang.org/grpc/credentials:all-srcs",
        "//vendor/google.golang.org/grpc/grpclog:all-srcs",
        "//vendor/google.golang.org/grpc/internal:all-srcs",
        "//vendor/google.golang.org/grpc/metadata:all-srcs",
        "//vendor/google.golang.org/grpc/naming:all-srcs",
        "//vendor/google.golang.org/grpc/peer:all-srcs",
        "//vendor/google.golang.org/grpc/transport:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
