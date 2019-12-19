load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "addr_linux.go",
        "bridge_linux.go",
        "conntrack_linux.go",
        "genetlink_linux.go",
        "link_linux.go",
        "mpls_linux.go",
        "nl_linux.go",
        "nl_unspecified.go",
        "route_linux.go",
        "seg6_linux.go",
        "syscall.go",
        "tc_linux.go",
        "xfrm_linux.go",
        "xfrm_monitor_linux.go",
        "xfrm_policy_linux.go",
        "xfrm_state_linux.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/vishvananda/netlink/nl",
    importpath = "github.com/vishvananda/netlink/nl",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/github.com/vishvananda/netns:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
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
