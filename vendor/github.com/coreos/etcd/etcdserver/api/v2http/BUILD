load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "capability.go",
        "client.go",
        "client_auth.go",
        "doc.go",
        "http.go",
        "metrics.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/coreos/etcd/etcdserver/api/v2http",
    importpath = "github.com/coreos/etcd/etcdserver/api/v2http",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/etcd/error:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/api:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/api/etcdhttp:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/api/v2http/httptypes:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/auth:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/etcdserverpb:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/membership:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/stats:go_default_library",
        "//vendor/github.com/coreos/etcd/pkg/logutil:go_default_library",
        "//vendor/github.com/coreos/etcd/pkg/types:go_default_library",
        "//vendor/github.com/coreos/etcd/store:go_default_library",
        "//vendor/github.com/coreos/pkg/capnslog:go_default_library",
        "//vendor/github.com/jonboulle/clockwork:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/coreos/etcd/etcdserver/api/v2http/httptypes:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
