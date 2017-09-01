package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["doc.go"],
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
        "//federation/pkg/federation-controller/cluster:all-srcs",
        "//federation/pkg/federation-controller/ingress:all-srcs",
        "//federation/pkg/federation-controller/job:all-srcs",
        "//federation/pkg/federation-controller/service:all-srcs",
        "//federation/pkg/federation-controller/sync:all-srcs",
        "//federation/pkg/federation-controller/util:all-srcs",
    ],
    tags = ["automanaged"],
)
