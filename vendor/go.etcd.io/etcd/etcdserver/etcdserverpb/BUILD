load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "etcdserver.pb.go",
        "raft_internal.pb.go",
        "raft_internal_stringer.go",
        "rpc.pb.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/etcdserver/etcdserverpb",
    importpath = "go.etcd.io/etcd/etcdserver/etcdserverpb",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/gogo/protobuf/gogoproto:go_default_library",
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
        "//vendor/go.etcd.io/etcd/auth/authpb:go_default_library",
        "//vendor/go.etcd.io/etcd/mvcc/mvccpb:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/google.golang.org/grpc:go_default_library",
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
        "//vendor/go.etcd.io/etcd/etcdserver/etcdserverpb/gw:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
