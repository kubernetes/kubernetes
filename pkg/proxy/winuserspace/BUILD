package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "loadbalancer.go",
        "port_allocator.go",
        "proxier.go",
        "proxysocket.go",
        "roundrobin.go",
        "udp_server.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/proxy:go_default_library",
        "//pkg/util/netsh:go_default_library",
        "//pkg/util/slice:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "port_allocator_test.go",
        "proxier_test.go",
        "roundrobin_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/proxy:go_default_library",
        "//pkg/util/netsh/testing:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
