load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "addr.go",
        "class.go",
        "filter.go",
        "fou.go",
        "link.go",
        "neigh.go",
        "netlink.go",
        "order.go",
        "protinfo.go",
        "qdisc.go",
        "route.go",
        "rule.go",
        "socket.go",
        "xfrm.go",
        "xfrm_policy.go",
        "xfrm_state.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "conntrack_unspecified.go",
            "fou_unspecified.go",
            "genetlink_unspecified.go",
            "handle_unspecified.go",
            "netlink_unspecified.go",
            "route_unspecified.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "conntrack_unspecified.go",
            "fou_unspecified.go",
            "genetlink_unspecified.go",
            "handle_unspecified.go",
            "netlink_unspecified.go",
            "route_unspecified.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "conntrack_unspecified.go",
            "fou_unspecified.go",
            "genetlink_unspecified.go",
            "handle_unspecified.go",
            "netlink_unspecified.go",
            "route_unspecified.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "conntrack_unspecified.go",
            "fou_unspecified.go",
            "genetlink_unspecified.go",
            "handle_unspecified.go",
            "netlink_unspecified.go",
            "route_unspecified.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "addr_linux.go",
            "bpf_linux.go",
            "bridge_linux.go",
            "class_linux.go",
            "conntrack_linux.go",
            "filter_linux.go",
            "fou_linux.go",
            "genetlink_linux.go",
            "gtp_linux.go",
            "handle_linux.go",
            "link_linux.go",
            "link_tuntap_linux.go",
            "neigh_linux.go",
            "netlink_linux.go",
            "protinfo_linux.go",
            "qdisc_linux.go",
            "route_linux.go",
            "rule_linux.go",
            "socket_linux.go",
            "xfrm_monitor_linux.go",
            "xfrm_policy_linux.go",
            "xfrm_state_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "conntrack_unspecified.go",
            "fou_unspecified.go",
            "genetlink_unspecified.go",
            "handle_unspecified.go",
            "netlink_unspecified.go",
            "route_unspecified.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "conntrack_unspecified.go",
            "fou_unspecified.go",
            "genetlink_unspecified.go",
            "handle_unspecified.go",
            "netlink_unspecified.go",
            "route_unspecified.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "conntrack_unspecified.go",
            "fou_unspecified.go",
            "genetlink_unspecified.go",
            "handle_unspecified.go",
            "netlink_unspecified.go",
            "route_unspecified.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "conntrack_unspecified.go",
            "fou_unspecified.go",
            "genetlink_unspecified.go",
            "handle_unspecified.go",
            "netlink_unspecified.go",
            "route_unspecified.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "conntrack_unspecified.go",
            "fou_unspecified.go",
            "genetlink_unspecified.go",
            "handle_unspecified.go",
            "netlink_unspecified.go",
            "route_unspecified.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "conntrack_unspecified.go",
            "fou_unspecified.go",
            "genetlink_unspecified.go",
            "handle_unspecified.go",
            "netlink_unspecified.go",
            "route_unspecified.go",
        ],
        "//conditions:default": [],
    }),
    cgo = True,
    importpath = "github.com/vishvananda/netlink",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/vishvananda/netlink/nl:go_default_library",
        "//vendor/golang.org/x/sys/unix:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/github.com/vishvananda/netns:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/github.com/vishvananda/netns:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/github.com/vishvananda/netns:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/github.com/vishvananda/netns:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/vishvananda/netns:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//vendor/github.com/vishvananda/netns:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/github.com/vishvananda/netns:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/github.com/vishvananda/netns:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/github.com/vishvananda/netns:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/github.com/vishvananda/netns:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/vishvananda/netns:go_default_library",
        ],
        "//conditions:default": [],
    }),
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
