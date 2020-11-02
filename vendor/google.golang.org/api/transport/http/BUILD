load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["dial.go"],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/api/transport/http",
    importpath = "google.golang.org/api/transport/http",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.opencensus.io/plugin/ochttp:go_default_library",
        "//vendor/golang.org/x/oauth2:go_default_library",
        "//vendor/google.golang.org/api/googleapi/transport:go_default_library",
        "//vendor/google.golang.org/api/internal:go_default_library",
        "//vendor/google.golang.org/api/option:go_default_library",
        "//vendor/google.golang.org/api/transport/http/internal/propagation:go_default_library",
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
        "//vendor/google.golang.org/api/transport/http/internal/propagation:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
