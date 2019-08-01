package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["keepalive.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/client-go/util/keepalive",
    importpath = "k8s.io/client-go/util/keepalive",
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
        "//staging/src/k8s.io/client-go/util/keepalive/http:all-srcs",
        "//staging/src/k8s.io/client-go/util/keepalive/restclient:all-srcs",
        "//staging/src/k8s.io/client-go/util/keepalive/spdy:all-srcs",
    ],
    tags = ["automanaged"],
)
