load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "syscall.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "nl_unspecified.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "nl_unspecified.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "nl_unspecified.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "nl_unspecified.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "addr_linux.go",
            "bridge_linux.go",
            "conntrack_linux.go",
            "genetlink_linux.go",
            "link_linux.go",
            "mpls_linux.go",
            "nl_linux.go",
            "route_linux.go",
            "seg6_linux.go",
            "tc_linux.go",
            "xfrm_linux.go",
            "xfrm_monitor_linux.go",
            "xfrm_policy_linux.go",
            "xfrm_state_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "nl_unspecified.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "nl_unspecified.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "nl_unspecified.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "nl_unspecified.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "nl_unspecified.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "nl_unspecified.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/vishvananda/netlink/nl",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/vishvananda/netns:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
