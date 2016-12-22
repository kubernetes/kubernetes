package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "node_conformance.go",
        "node_e2e.go",
        "remote.go",
        "ssh.go",
        "types.go",
        "utils.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/util/errors:go_default_library",
        "//test/e2e_node/builder:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
