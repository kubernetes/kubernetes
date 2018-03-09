package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

filegroup(
    name = "config",
    srcs = glob([
        "**/*.yaml",
        "**/*.yml",
        "**/*.json",
    ]) + [
        "pod",
    ],
)

go_library(
    name = "go_default_library",
    srcs = ["doc.go"],
    importpath = "k8s.io/kubernetes/examples",
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
        "//examples/explorer:all-srcs",
        "//examples/guestbook-go:all-srcs",
        "//examples/https-nginx:all-srcs",
        "//examples/sharing-clusters:all-srcs",
    ],
    tags = ["automanaged"],
)
