load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

licenses(["notice"])

go_library(
    name = "go_default_library",
    srcs = [
        "constants.go",
        "ipvs.go",
        "netlink.go",
    ],
    importpath = "k8s.io/kubernetes/third_party/forked/ipvs",
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

go_test(
    name = "go_default_test",
    srcs = ["netlink_test.go"],
    embed = [":go_default_library"],
)
