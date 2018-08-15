load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "client.go",
        "cluster.go",
        "node.go",
        "volume.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/sigma/heketi/client/api/go-client",
    importpath = "github.com/sigma/heketi/client/api/go-client",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/dgrijalva/jwt-go:go_default_library",
        "//vendor/github.com/sigma/heketi/pkg/glusterfs/api:go_default_library",
        "//vendor/github.com/sigma/heketi/pkg/utils:go_default_library",
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
    visibility = ["//visibility:public"],
)
