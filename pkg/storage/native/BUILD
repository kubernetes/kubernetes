package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "bitset.go",
        "bucket.go",
        "grpc_watcher.go",
        "itemdata.go",
        "log.go",
        "native.pb.go",
        "raft_backend.go",
        "raft_fsm.go",
        "server.go",
        "store.go",
        "ttl_queue.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/meta:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/conversion:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/storage:go_default_library",
        "//pkg/storage/etcd:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/gogo/protobuf/io",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/hashicorp/raft",
        "//vendor:github.com/hashicorp/raft-boltdb",
        "//vendor:golang.org/x/net/context",
        "//vendor:google.golang.org/grpc",
    ],
)
