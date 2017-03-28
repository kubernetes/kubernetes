package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "kube-proxy",
    library = ":go_default_library",
    tags = ["automanaged"],
)

go_library(
    name = "go_default_library",
    srcs = ["proxy.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/kube-proxy/app:go_default_library",
        "//cmd/kube-proxy/app/options:go_default_library",
        "//pkg/client/metrics/prometheus:go_default_library",
        "//pkg/version/prometheus:go_default_library",
        "//pkg/version/verflag:go_default_library",
        "//vendor:github.com/spf13/pflag",
        "//vendor:k8s.io/apiserver/pkg/server/healthz",
        "//vendor:k8s.io/apiserver/pkg/util/flag",
        "//vendor:k8s.io/apiserver/pkg/util/logs",
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
        "//cmd/kube-proxy/app:all-srcs",
    ],
    tags = ["automanaged"],
)
