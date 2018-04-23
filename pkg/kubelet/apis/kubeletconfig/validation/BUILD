package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "validation.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "validation_others.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "validation_others.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "validation_others.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "validation_others.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "validation_others.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "validation_others.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "validation_others.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "validation_others.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "validation_others.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "validation_others.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "validation_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/validation",
    deps = [
        "//pkg/features:go_default_library",
        "//pkg/kubelet/apis/kubeletconfig:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/validation:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/feature:go_default_library",
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
    srcs = ["validation_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/kubelet/apis/kubeletconfig:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
    ],
)
