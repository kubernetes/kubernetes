load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "client.go",
        "client_metrics.go",
        "client_stats_handler.go",
        "doc.go",
        "server.go",
        "server_metrics.go",
        "server_stats_handler.go",
        "stats_common.go",
        "trace_common.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.opencensus.io/plugin/ocgrpc",
    importpath = "go.opencensus.io/plugin/ocgrpc",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.opencensus.io/stats:go_default_library",
        "//vendor/go.opencensus.io/stats/view:go_default_library",
        "//vendor/go.opencensus.io/tag:go_default_library",
        "//vendor/go.opencensus.io/trace:go_default_library",
        "//vendor/go.opencensus.io/trace/propagation:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/google.golang.org/grpc/codes:go_default_library",
        "//vendor/google.golang.org/grpc/grpclog:go_default_library",
        "//vendor/google.golang.org/grpc/metadata:go_default_library",
        "//vendor/google.golang.org/grpc/stats:go_default_library",
        "//vendor/google.golang.org/grpc/status:go_default_library",
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
