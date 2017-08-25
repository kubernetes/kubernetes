package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "types.go",
    ],
    deps = ["//vendor/k8s.io/apimachinery/pkg/types:go_default_library"],
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
        "//pkg/proxy/config:all-srcs",
        "//pkg/proxy/healthcheck:all-srcs",
        "//pkg/proxy/iptables:all-srcs",
        "//pkg/proxy/ipvs:all-srcs",
        "//pkg/proxy/userspace:all-srcs",
        "//pkg/proxy/util:all-srcs",
        "//pkg/proxy/winkernel:all-srcs",
        "//pkg/proxy/winuserspace:all-srcs",
    ],
    tags = ["automanaged"],
)
