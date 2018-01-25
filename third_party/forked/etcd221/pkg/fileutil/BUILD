package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "fileutil_test.go",
        "lock_test.go",
        "preallocate_test.go",
        "purge_test.go",
    ],
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/third_party/forked/etcd221/pkg/fileutil",
)

go_library(
    name = "go_default_library",
    srcs = [
        "fileutil.go",
        "purge.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "lock_unix.go",
            "perallocate_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "lock_unix.go",
            "perallocate_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "lock_unix.go",
            "perallocate_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "lock_unix.go",
            "perallocate_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "lock_unix.go",
            "preallocate.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "lock_unix.go",
            "perallocate_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "lock_unix.go",
            "perallocate_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "lock_unix.go",
            "perallocate_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "lock_plan9.go",
            "perallocate_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "lock_solaris.go",
            "perallocate_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "lock_windows.go",
            "perallocate_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/third_party/forked/etcd221/pkg/fileutil",
    deps = [
        "//vendor/github.com/coreos/pkg/capnslog:go_default_library",
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
