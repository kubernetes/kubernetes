load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bdp_estimator.go",
        "controlbuf.go",
        "flowcontrol.go",
        "go16.go",
        "go17.go",
        "handler_server.go",
        "http2_client.go",
        "http2_server.go",
        "http_util.go",
        "log.go",
        "transport.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/grpc/transport",
    importpath = "google.golang.org/grpc/transport",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/golang.org/x/net/http2:go_default_library",
        "//vendor/golang.org/x/net/http2/hpack:go_default_library",
        "//vendor/google.golang.org/genproto/googleapis/rpc/status:go_default_library",
        "//vendor/google.golang.org/grpc/codes:go_default_library",
        "//vendor/google.golang.org/grpc/credentials:go_default_library",
        "//vendor/google.golang.org/grpc/grpclog:go_default_library",
        "//vendor/google.golang.org/grpc/internal/channelz:go_default_library",
        "//vendor/google.golang.org/grpc/internal/grpcrand:go_default_library",
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
