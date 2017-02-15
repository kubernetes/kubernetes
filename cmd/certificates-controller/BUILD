package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["certificates-controller.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/certificates-controller/app:go_default_library",
        "//pkg/util/logs:go_default_library",
        "//pkg/version/verflag:go_default_library",
        "//vendor:github.com/spf13/pflag",
        "//vendor:k8s.io/apiserver/pkg/util/flag",
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
        "//cmd/certificates-controller/app:all-srcs",
    ],
    tags = ["automanaged"],
)

go_binary(
    name = "certificates-controller",
    library = ":go_default_library",
    tags = ["automanaged"],
)
