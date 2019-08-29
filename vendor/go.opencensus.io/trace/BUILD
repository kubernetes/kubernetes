load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "basetypes.go",
        "config.go",
        "doc.go",
        "evictedqueue.go",
        "export.go",
        "lrumap.go",
        "sampling.go",
        "spanbucket.go",
        "spanstore.go",
        "status_codes.go",
        "trace.go",
        "trace_go11.go",
        "trace_nongo11.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.opencensus.io/trace",
    importpath = "go.opencensus.io/trace",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/hashicorp/golang-lru/simplelru:go_default_library",
        "//vendor/go.opencensus.io/internal:go_default_library",
        "//vendor/go.opencensus.io/trace/internal:go_default_library",
        "//vendor/go.opencensus.io/trace/tracestate:go_default_library",
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
        "//vendor/go.opencensus.io/trace/internal:all-srcs",
        "//vendor/go.opencensus.io/trace/propagation:all-srcs",
        "//vendor/go.opencensus.io/trace/tracestate:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
