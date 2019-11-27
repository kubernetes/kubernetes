load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bdp_estimator.go",
        "controlbuf.go",
        "defaults.go",
        "flowcontrol.go",
        "handler_server.go",
        "http2_client.go",
        "http2_server.go",
        "http_util.go",
        "log.go",
        "transport.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/grpc/internal/transport",
    importpath = "google.golang.org/grpc/internal/transport",
    visibility = ["//vendor/google.golang.org/grpc:__subpackages__"],
    deps = [
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
        "//vendor/golang.org/x/net/http2:go_default_library",
        "//vendor/golang.org/x/net/http2/hpack:go_default_library",
        "//vendor/google.golang.org/genproto/googleapis/rpc/status:go_default_library",
        "//vendor/google.golang.org/grpc/codes:go_default_library",
        "//vendor/google.golang.org/grpc/credentials:go_default_library",
        "//vendor/google.golang.org/grpc/grpclog:go_default_library",
        "//vendor/google.golang.org/grpc/internal:go_default_library",
        "//vendor/google.golang.org/grpc/internal/channelz:go_default_library",
        "//vendor/google.golang.org/grpc/internal/grpcrand:go_default_library",
        "//vendor/google.golang.org/grpc/internal/syscall:go_default_library",
        "//vendor/google.golang.org/grpc/keepalive:go_default_library",
        "//vendor/google.golang.org/grpc/metadata:go_default_library",
        "//vendor/google.golang.org/grpc/peer:go_default_library",
        "//vendor/google.golang.org/grpc/stats:go_default_library",
        "//vendor/google.golang.org/grpc/status:go_default_library",
        "//vendor/google.golang.org/grpc/tap:go_default_library",
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
