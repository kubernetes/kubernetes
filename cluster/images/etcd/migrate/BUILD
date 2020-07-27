load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "copy_file.go",
        "data_dir.go",
        "migrate.go",
        "migrate_client.go",
        "migrate_server.go",
        "migrator.go",
        "options.go",
        "util_others.go",
        "utils_windows.go",
        "versions.go",
    ],
    importpath = "k8s.io/kubernetes/cluster/images/etcd/migrate",
    visibility = ["//visibility:private"],
    deps = [
        "//vendor/github.com/blang/semver:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/go.etcd.io/etcd/client:go_default_library",
        "//vendor/go.etcd.io/etcd/clientv3:go_default_library",
        "//vendor/google.golang.org/grpc:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:aix": [
            "//vendor/github.com/mrunalp/fileutils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/github.com/mrunalp/fileutils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/github.com/mrunalp/fileutils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/github.com/mrunalp/fileutils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/github.com/mrunalp/fileutils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:illumos": [
            "//vendor/github.com/mrunalp/fileutils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:ios": [
            "//vendor/github.com/mrunalp/fileutils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:js": [
            "//vendor/github.com/mrunalp/fileutils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/mrunalp/fileutils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//vendor/github.com/mrunalp/fileutils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/github.com/mrunalp/fileutils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/github.com/mrunalp/fileutils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/github.com/mrunalp/fileutils:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/github.com/mrunalp/fileutils:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_binary(
    name = "migrate",
    embed = [":go_default_library"],
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "data_dir_test.go",
        "options_test.go",
        "versions_test.go",
    ],
    data = glob(["testdata/**"]),
    embed = [":go_default_library"],
    deps = ["//vendor/github.com/blang/semver:go_default_library"],
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
