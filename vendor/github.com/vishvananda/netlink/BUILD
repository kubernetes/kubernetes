load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "addr.go",
        "filter.go",
        "link.go",
        "neigh.go",
        "netlink.go",
        "netlink_unspecified.go",
        "protinfo.go",
        "qdisc.go",
        "route.go",
        "xfrm.go",
        "xfrm_policy.go",
        "xfrm_state.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "addr_linux.go",
            "filter_linux.go",
            "link_linux.go",
            "neigh_linux.go",
            "protinfo_linux.go",
            "qdisc_linux.go",
            "route_linux.go",
            "xfrm_policy_linux.go",
            "xfrm_state_linux.go",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
)
