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
    srcs = ["agnhost.go"],
    importpath = "k8s.io/kubernetes/test/images/agnhost",
    deps = [
        "//test/images/agnhost/dns:go_default_library",
        "//test/images/agnhost/fakegitserver:go_default_library",
        "//test/images/agnhost/liveness:go_default_library",
        "//test/images/agnhost/logs-generator:go_default_library",
        "//test/images/agnhost/net:go_default_library",
        "//test/images/agnhost/netexec:go_default_library",
        "//test/images/agnhost/nettest:go_default_library",
        "//test/images/agnhost/no-snat-test:go_default_library",
        "//test/images/agnhost/no-snat-test-proxy:go_default_library",
        "//test/images/agnhost/pause:go_default_library",
        "//test/images/agnhost/port-forward-tester:go_default_library",
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
        "//test/images/agnhost/dns:all-srcs",
        "//test/images/agnhost/fakegitserver:all-srcs",
        "//test/images/agnhost/liveness:all-srcs",
        "//test/images/agnhost/logs-generator:all-srcs",
        "//test/images/agnhost/net:all-srcs",
        "//test/images/agnhost/netexec:all-srcs",
        "//test/images/agnhost/nettest:all-srcs",
        "//test/images/agnhost/no-snat-test:all-srcs",
        "//test/images/agnhost/no-snat-test-proxy:all-srcs",
        "//test/images/agnhost/pause:all-srcs",
        "//test/images/agnhost/port-forward-tester:all-srcs",
        "//test/images/agnhost/webhook:all-srcs",
    ],
    tags = ["automanaged"],
)
