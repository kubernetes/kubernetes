package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["doc.go"],
    tags = ["automanaged"],
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
        "//federation/pkg/federation-controller/configmap:all-srcs",
        "//federation/pkg/federation-controller/daemonset:all-srcs",
        "//federation/pkg/federation-controller/deployment:all-srcs",
        "//federation/pkg/federation-controller/ingress:all-srcs",
        "//federation/pkg/federation-controller/namespace:all-srcs",
        "//federation/pkg/federation-controller/replicaset:all-srcs",
        "//federation/pkg/federation-controller/secret:all-srcs",
        "//federation/pkg/federation-controller/service:all-srcs",
        "//federation/pkg/federation-controller/util:all-srcs",
    ],
    tags = ["automanaged"],
)
