load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "volume_path_handler.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "volume_path_handler_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "volume_path_handler_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "volume_path_handler_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "volume_path_handler_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "volume_path_handler_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "volume_path_handler_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "volume_path_handler_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "volume_path_handler_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "volume_path_handler_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "volume_path_handler_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "volume_path_handler_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/volume/util/volumepathhandler",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
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
    visibility = ["//visibility:public"],
)
