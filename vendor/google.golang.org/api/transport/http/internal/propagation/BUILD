load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["http.go"],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/api/transport/http/internal/propagation",
    importpath = "google.golang.org/api/transport/http/internal/propagation",
    visibility = ["//vendor/google.golang.org/api/transport/http:__subpackages__"],
    deps = [
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
