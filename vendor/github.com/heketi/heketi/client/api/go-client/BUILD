load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "admin.go",
        "backup.go",
        "block_volume.go",
        "client.go",
        "cluster.go",
        "db.go",
        "device.go",
        "logging.go",
        "node.go",
        "operations.go",
        "state.go",
        "topology.go",
        "volume.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/heketi/heketi/client/api/go-client",
    importpath = "github.com/heketi/heketi/client/api/go-client",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/dgrijalva/jwt-go:go_default_library",
        "//vendor/github.com/heketi/heketi/pkg/glusterfs/api:go_default_library",
        "//vendor/github.com/heketi/heketi/pkg/utils:go_default_library",
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
