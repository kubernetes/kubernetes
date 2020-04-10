load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "delegator.go",
        "http.go",
        "instrument_client.go",
        "instrument_server.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/prometheus/client_golang/prometheus/promhttp",
    importpath = "github.com/prometheus/client_golang/prometheus/promhttp",
    visibility = ["//build/visible_to:vendor_githubcom_prometheus_CONSUMERS"],
    deps = [
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/github.com/prometheus/client_model/go:go_default_library",
        "//vendor/github.com/prometheus/common/expfmt:go_default_library",
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
