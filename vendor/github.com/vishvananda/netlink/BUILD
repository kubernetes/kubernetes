package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "addr.go",
        "addr_linux.go",
        "filter.go",
        "filter_linux.go",
        "link.go",
        "link_linux.go",
        "neigh.go",
        "neigh_linux.go",
        "netlink.go",
        "protinfo.go",
        "protinfo_linux.go",
        "qdisc.go",
        "qdisc_linux.go",
        "route.go",
        "route_linux.go",
        "xfrm.go",
        "xfrm_policy.go",
        "xfrm_policy_linux.go",
        "xfrm_state.go",
        "xfrm_state_linux.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor/github.com/vishvananda/netlink/nl:go_default_library"],
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
        "//vendor/github.com/vishvananda/netlink/nl:all-srcs",
    ],
    tags = ["automanaged"],
)
