load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "constants.go",
        "ipvs.go",
        "netlink.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/libnetwork/ipvs",
    importpath = "github.com/docker/libnetwork/ipvs",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/github.com/sirupsen/logrus:go_default_library",
            "//vendor/github.com/vishvananda/netlink/nl:go_default_library",
            "//vendor/github.com/vishvananda/netns:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/sirupsen/logrus:go_default_library",
            "//vendor/github.com/vishvananda/netlink/nl:go_default_library",
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
