load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "nl_unspecified.go",
        "syscall.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "addr_linux.go",
            "bridge_linux.go",
            "conntrack_linux.go",
            "genetlink_linux.go",
            "link_linux.go",
            "mpls_linux.go",
            "nl_linux.go",
            "route_linux.go",
            "tc_linux.go",
            "xfrm_linux.go",
            "xfrm_monitor_linux.go",
            "xfrm_policy_linux.go",
            "xfrm_state_linux.go",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
