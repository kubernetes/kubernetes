load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "discard_logger.go",
        "doc.go",
        "log_level.go",
        "logger.go",
        "merge_logger.go",
        "package_logger.go",
        "zap.go",
        "zap_grpc.go",
        "zap_journal.go",
        "zap_raft.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.etcd.io/etcd/pkg/logutil",
    importpath = "go.etcd.io/etcd/pkg/logutil",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/pkg/capnslog:go_default_library",
        "//vendor/go.etcd.io/etcd/raft:go_default_library",
        "//vendor/go.uber.org/zap:go_default_library",
        "//vendor/go.uber.org/zap/zapcore:go_default_library",
        "//vendor/google.golang.org/grpc/grpclog:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:aix": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
            "//vendor/go.etcd.io/etcd/pkg/systemd:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
            "//vendor/go.etcd.io/etcd/pkg/systemd:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
            "//vendor/go.etcd.io/etcd/pkg/systemd:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
            "//vendor/go.etcd.io/etcd/pkg/systemd:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
            "//vendor/go.etcd.io/etcd/pkg/systemd:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:illumos": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
            "//vendor/go.etcd.io/etcd/pkg/systemd:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:ios": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
            "//vendor/go.etcd.io/etcd/pkg/systemd:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:js": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
            "//vendor/go.etcd.io/etcd/pkg/systemd:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
            "//vendor/go.etcd.io/etcd/pkg/systemd:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
            "//vendor/go.etcd.io/etcd/pkg/systemd:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
            "//vendor/go.etcd.io/etcd/pkg/systemd:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
            "//vendor/go.etcd.io/etcd/pkg/systemd:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
            "//vendor/go.etcd.io/etcd/pkg/systemd:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/github.com/coreos/go-systemd/journal:go_default_library",
            "//vendor/go.etcd.io/etcd/pkg/systemd:go_default_library",
        ],
        "//conditions:default": [],
    }),
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
