load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "auth_role.go",
        "auth_user.go",
        "cancelreq.go",
        "client.go",
        "cluster_error.go",
        "curl.go",
        "discover.go",
        "doc.go",
        "json.go",
        "keys.go",
        "members.go",
        "util.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/coreos/etcd/client",
    importpath = "github.com/coreos/etcd/client",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/etcd/pkg/pathutil:go_default_library",
        "//vendor/github.com/coreos/etcd/pkg/srv:go_default_library",
        "//vendor/github.com/coreos/etcd/pkg/types:go_default_library",
        "//vendor/github.com/coreos/etcd/version:go_default_library",
        "//vendor/github.com/json-iterator/go:go_default_library",
        "//vendor/github.com/modern-go/reflect2:go_default_library",
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
