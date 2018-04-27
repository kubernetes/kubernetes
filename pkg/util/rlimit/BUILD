package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = select({
        "@io_bazel_rules_go//go/platform:android": [
            "rlimit_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "rlimit_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "rlimit_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "rlimit_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "rlimit_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "rlimit_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "rlimit_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "rlimit_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "rlimit_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "rlimit_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "rlimit_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/util/rlimit",
    deps = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
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
)
