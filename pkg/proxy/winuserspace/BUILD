package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
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
        "//pkg/api:go_default_library",
        "//pkg/proxy:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//pkg/util/net:go_default_library",
        "//pkg/util/netsh:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/util/slice:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "port_allocator_test.go",
        "proxier_test.go",
        "roundrobin_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/proxy:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/net:go_default_library",
        "//pkg/util/netsh/testing:go_default_library",
        "//pkg/util/runtime:go_default_library",
    ],
)
