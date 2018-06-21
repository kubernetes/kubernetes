load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "client.go",
        "generatexid.go",
        "inetsock.go",
        "pktsock_linux.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/d2g/dhcp4client",
    importpath = "github.com/d2g/dhcp4client",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/d2g/dhcp4:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
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
