load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "client.go",
        "client_stats.go",
        "doc.go",
        "route.go",
        "server.go",
        "span_annotating_client_trace.go",
        "stats.go",
        "trace.go",
        "wrapped_body.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.opencensus.io/plugin/ochttp",
    importpath = "go.opencensus.io/plugin/ochttp",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.opencensus.io/plugin/ochttp/propagation/b3:go_default_library",
        "//vendor/go.opencensus.io/stats:go_default_library",
        "//vendor/go.opencensus.io/stats/view:go_default_library",
        "//vendor/go.opencensus.io/tag:go_default_library",
        "//vendor/go.opencensus.io/trace:go_default_library",
        "//vendor/go.opencensus.io/trace/propagation:go_default_library",
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
        "//vendor/go.opencensus.io/plugin/ochttp/propagation/b3:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
