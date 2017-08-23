load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["handlers.go"],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/abbot/go-http-auth:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/google/cadvisor/api:go_default_library",
        "//vendor/github.com/google/cadvisor/healthz:go_default_library",
        "//vendor/github.com/google/cadvisor/http/mux:go_default_library",
        "//vendor/github.com/google/cadvisor/manager:go_default_library",
        "//vendor/github.com/google/cadvisor/metrics:go_default_library",
        "//vendor/github.com/google/cadvisor/pages:go_default_library",
        "//vendor/github.com/google/cadvisor/pages/static:go_default_library",
        "//vendor/github.com/google/cadvisor/validate:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus/promhttp:go_default_library",
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
        "//vendor/github.com/google/cadvisor/http/mux:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
