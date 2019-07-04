load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "addr.go",
        "addr_linux.go",
        "bpf_linux.go",
        "bridge_linux.go",
        "class.go",
        "class_linux.go",
        "conntrack_linux.go",
        "conntrack_unspecified.go",
        "filter.go",
        "filter_linux.go",
        "genetlink_linux.go",
        "genetlink_unspecified.go",
        "gtp_linux.go",
        "handle_linux.go",
        "handle_unspecified.go",
        "link.go",
        "link_linux.go",
        "link_tuntap_linux.go",
        "neigh.go",
        "neigh_linux.go",
        "netlink.go",
        "netlink_linux.go",
        "netlink_unspecified.go",
        "order.go",
        "protinfo.go",
        "protinfo_linux.go",
        "qdisc.go",
        "qdisc_linux.go",
        "route.go",
        "route_linux.go",
        "route_unspecified.go",
        "rule.go",
        "rule_linux.go",
        "socket.go",
        "socket_linux.go",
        "xfrm.go",
        "xfrm_monitor_linux.go",
        "xfrm_policy.go",
        "xfrm_policy_linux.go",
        "xfrm_state.go",
        "xfrm_state_linux.go",
    ],
    cgo = True,
    importmap = "k8s.io/kubernetes/vendor/github.com/vishvananda/netlink",
    importpath = "github.com/vishvananda/netlink",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/vishvananda/netlink/nl:go_default_library",
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
