load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "jwt.go",
        "nop.go",
        "options.go",
        "range_perm_cache.go",
        "simple_token.go",
        "store.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/auth",
    importpath = "go.etcd.io/etcd/auth",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/pkg/capnslog:go_default_library",
        "//vendor/github.com/dgrijalva/jwt-go:go_default_library",
        "//vendor/go.etcd.io/etcd/auth/authpb:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/api/v3rpc/rpctypes:go_default_library",
        "//vendor/go.etcd.io/etcd/etcdserver/etcdserverpb:go_default_library",
        "//vendor/go.etcd.io/etcd/mvcc/backend:go_default_library",
        "//vendor/go.etcd.io/etcd/pkg/adt:go_default_library",
        "//vendor/go.uber.org/zap:go_default_library",
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
        "//vendor/go.etcd.io/etcd/auth/authpb:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
