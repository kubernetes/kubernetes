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
        "cloudstack.go",
        "cloudstack_loadbalancer.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/xanzy/go-cloudstack/cloudstack",
        "//vendor:gopkg.in/gcfg.v1",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["cloudstack_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = ["//pkg/api:go_default_library"],
)
