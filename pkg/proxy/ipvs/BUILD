package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "iptables.go",
        "proxier.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/proxy:go_default_library",
        "//pkg/proxy/ipvs/ipvsclient:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//pkg/util/iptables:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/slice:go_default_library",
        "//vendor:github.com/davecgh/go-spew/spew",
        "//vendor:github.com/golang/glog",
    ],
)
