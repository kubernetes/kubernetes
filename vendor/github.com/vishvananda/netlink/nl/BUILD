package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "addr_linux.go",
        "link_linux.go",
        "nl_linux.go",
        "route_linux.go",
        "tc_linux.go",
        "xfrm_linux.go",
        "xfrm_policy_linux.go",
        "xfrm_state_linux.go",
    ],
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
