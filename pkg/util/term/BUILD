package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = select({
        "@io_bazel_rules_go//go/platform:android": [
            "setsize.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "setsize.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "setsize.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "setsize.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "setsize.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "setsize.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "setsize.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "setsize.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "setsize.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "setsize.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "setsize_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/util/term",
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/github.com/docker/docker/pkg/term:go_default_library",
            "//vendor/k8s.io/client-go/tools/remotecommand:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/github.com/docker/docker/pkg/term:go_default_library",
            "//vendor/k8s.io/client-go/tools/remotecommand:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/github.com/docker/docker/pkg/term:go_default_library",
            "//vendor/k8s.io/client-go/tools/remotecommand:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/github.com/docker/docker/pkg/term:go_default_library",
            "//vendor/k8s.io/client-go/tools/remotecommand:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/docker/docker/pkg/term:go_default_library",
            "//vendor/k8s.io/client-go/tools/remotecommand:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//vendor/github.com/docker/docker/pkg/term:go_default_library",
            "//vendor/k8s.io/client-go/tools/remotecommand:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/github.com/docker/docker/pkg/term:go_default_library",
            "//vendor/k8s.io/client-go/tools/remotecommand:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/github.com/docker/docker/pkg/term:go_default_library",
            "//vendor/k8s.io/client-go/tools/remotecommand:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/github.com/docker/docker/pkg/term:go_default_library",
            "//vendor/k8s.io/client-go/tools/remotecommand:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/github.com/docker/docker/pkg/term:go_default_library",
            "//vendor/k8s.io/client-go/tools/remotecommand:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/k8s.io/client-go/tools/remotecommand:go_default_library",
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
