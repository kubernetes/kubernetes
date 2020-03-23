package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["main.go"],
    importpath = "k8s.io/kubernetes/test/images/agnhost/net",
    deps = [
        "//test/images/agnhost/net/common:go_default_library",
        "//test/images/agnhost/net/nat:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
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
        "//test/images/agnhost/net/common:all-srcs",
        "//test/images/agnhost/net/nat:all-srcs",
    ],
    tags = ["automanaged"],
)
