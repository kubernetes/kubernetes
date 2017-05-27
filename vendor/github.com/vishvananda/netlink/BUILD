package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "cgo_genrule",
    "go_library",
)

cgo_genrule(
    name = "cgo_codegen",
    srcs = ["bpf_linux.go"],
    clinkopts = [
        "-lz",
        "-lm",
        "-lpthread",
        "-ldl",
    ],
    tags = ["automanaged"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "addr.go",
        "addr_linux.go",
        "class.go",
        "class_linux.go",
        "conntrack_linux.go",
        "filter.go",
        "filter_linux.go",
        "genetlink_linux.go",
        "gtp_linux.go",
        "handle_linux.go",
        "link.go",
        "link_linux.go",
        "link_tuntap_linux.go",
        "neigh.go",
        "neigh_linux.go",
        "netlink.go",
        "netlink_linux.go",
        "order.go",
        "protinfo.go",
        "protinfo_linux.go",
        "qdisc.go",
        "qdisc_linux.go",
        "route.go",
        "route_linux.go",
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
    library = ":cgo_codegen",
    tags = ["automanaged"],
    deps = [
        "//vendor/github.com/vishvananda/netlink/nl:go_default_library",
        "//vendor/github.com/vishvananda/netns:go_default_library",
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
        "//vendor/github.com/vishvananda/netlink/nl:all-srcs",
    ],
    tags = ["automanaged"],
)
