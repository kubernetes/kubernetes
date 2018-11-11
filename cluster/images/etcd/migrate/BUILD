package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
)

go_binary(
    name = "migrate",
    embed = [":go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "data_dir.go",
        "migrate.go",
        "migrate_client.go",
        "migrate_server.go",
        "migrator.go",
        "rollback_v2.go",
        "versions.go",
    ],
    importpath = "k8s.io/kubernetes/cluster/images/etcd/migrate",
    deps = [
        "//third_party/forked/etcd221/wal:go_default_library",
        "//vendor/github.com/blang/semver:go_default_library",
        "//vendor/github.com/coreos/etcd/client:go_default_library",
        "//vendor/github.com/coreos/etcd/clientv3:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/etcdserverpb:go_default_library",
        "//vendor/github.com/coreos/etcd/etcdserver/membership:go_default_library",
        "//vendor/github.com/coreos/etcd/mvcc/backend:go_default_library",
        "//vendor/github.com/coreos/etcd/mvcc/mvccpb:go_default_library",
        "//vendor/github.com/coreos/etcd/pkg/pbutil:go_default_library",
        "//vendor/github.com/coreos/etcd/pkg/types:go_default_library",
        "//vendor/github.com/coreos/etcd/raft/raftpb:go_default_library",
        "//vendor/github.com/coreos/etcd/snap:go_default_library",
        "//vendor/github.com/coreos/etcd/store:go_default_library",
        "//vendor/github.com/coreos/etcd/wal:go_default_library",
        "//vendor/github.com/coreos/etcd/wal/walpb:go_default_library",
        "//vendor/github.com/coreos/go-semver/semver:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
)

go_test(
    name = "go_default_test",
    srcs = [
        "data_dir_test.go",
        "versions_test.go",
    ],
    data = glob(["testdata/**"]),
    embed = [":go_default_library"],
    deps = ["//vendor/github.com/blang/semver:go_default_library"],
)
