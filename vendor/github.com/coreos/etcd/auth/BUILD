load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "jwt.go",
        "nop.go",
        "range_perm_cache.go",
        "simple_token.go",
        "store.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/coreos/etcd/auth",
    importpath = "github.com/coreos/etcd/auth",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/etcd/auth/authpb:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/etcdserverpb:go_default_library",
        "//vendor/github.com/coreos/etcd/mvcc/backend:go_default_library",
        "//vendor/github.com/coreos/etcd/pkg/adt:go_default_library",
        "//vendor/github.com/coreos/pkg/capnslog:go_default_library",
        "//vendor/github.com/dgrijalva/jwt-go:go_default_library",
        "//vendor/golang.org/x/crypto/bcrypt:go_default_library",
        "//vendor/google.golang.org/grpc/credentials:go_default_library",
        "//vendor/google.golang.org/grpc/metadata:go_default_library",
        "//vendor/google.golang.org/grpc/peer:go_default_library",
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
        "//vendor/github.com/coreos/etcd/auth/authpb:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
