load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "client.go",
        "client_metrics.go",
        "client_reporter.go",
        "metric_options.go",
        "server.go",
        "server_metrics.go",
        "server_reporter.go",
        "util.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/grpc-ecosystem/go-grpc-prometheus",
    importpath = "github.com/grpc-ecosystem/go-grpc-prometheus",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/google.golang.org/grpc:go_default_library",
        "//vendor/google.golang.org/grpc/codes:go_default_library",
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
