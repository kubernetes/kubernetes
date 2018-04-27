package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = select({
        "@io_bazel_rules_go//go/platform:android": [
            "flock_other.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "flock_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "flock_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "flock_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "flock_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "flock_other.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "flock_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "flock_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "flock_other.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "flock_other.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "flock_other.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/util/flock",
    deps = select({
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
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
