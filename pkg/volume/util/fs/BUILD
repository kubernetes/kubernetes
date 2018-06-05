load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = select({
        "@io_bazel_rules_go//go/platform:android": [
            "fs_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "fs.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "fs_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "fs_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "fs.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "fs_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "fs_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "fs_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "fs_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "fs_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "fs_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/volume/util/fs",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/golang.org/x/sys/windows:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
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
