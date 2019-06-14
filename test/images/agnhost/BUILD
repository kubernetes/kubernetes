package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "agnhost",
    embed = [":go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "agnhost.go",
        "common.go",
        "utils.go",
        "utils_windows.go",
    ],
    importpath = "k8s.io/kubernetes/test/images/agnhost",
    deps = [
        "//test/images/agnhost/net:go_default_library",
        "//test/images/agnhost/netexec:go_default_library",
        "//test/images/agnhost/nettest:go_default_library",
        "//test/images/agnhost/webhook:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
        "//test/images/agnhost/net:all-srcs",
        "//test/images/agnhost/netexec:all-srcs",
        "//test/images/agnhost/nettest:all-srcs",
        "//test/images/agnhost/webhook:all-srcs",
    ],
    tags = ["automanaged"],
)
