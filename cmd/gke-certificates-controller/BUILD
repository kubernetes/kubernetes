package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["main.go"],
    deps = [
        "//cmd/gke-certificates-controller/app:go_default_library",
        "//pkg/kubectl/util/logs:go_default_library",
        "//pkg/version/verflag:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/flag:go_default_library",
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
        "//cmd/gke-certificates-controller/app:all-srcs",
    ],
    tags = ["automanaged"],
)

go_binary(
    name = "gke-certificates-controller",
    library = ":go_default_library",
)
