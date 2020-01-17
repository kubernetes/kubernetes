load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "pidlimit.go",
        "pidlimit_linux.go",
        "pidlimit_unsupported.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/stats/pidlimit",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//pkg/kubelet/apis/stats/v1alpha1:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//pkg/kubelet/apis/stats/v1alpha1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//pkg/kubelet/apis/stats/v1alpha1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//pkg/kubelet/apis/stats/v1alpha1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:ios": [
            "//pkg/kubelet/apis/stats/v1alpha1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//pkg/kubelet/apis/stats/v1alpha1:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//pkg/kubelet/apis/stats/v1alpha1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//pkg/kubelet/apis/stats/v1alpha1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//pkg/kubelet/apis/stats/v1alpha1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//pkg/kubelet/apis/stats/v1alpha1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//pkg/kubelet/apis/stats/v1alpha1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//pkg/kubelet/apis/stats/v1alpha1:go_default_library",
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
