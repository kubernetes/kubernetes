load("@io_bazel_rules_go//go:def.bzl", "go_library")

filegroup(
    name = "go_default_library_protos",
    srcs = ["v3lock.proto"],
    visibility = ["//visibility:public"],
)

go_library(
    name = "go_default_library",
    srcs = ["v3lock.pb.go"],
    importpath = "github.com/coreos/etcd/etcdserver/api/v3lock/v3lockpb",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/etcd/etcdserver/etcdserverpb:go_default_library",
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/google.golang.org/genproto/googleapis/api/annotations:go_default_library",
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
        "//vendor/github.com/coreos/etcd/etcdserver/api/v3lock/v3lockpb/gw:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
