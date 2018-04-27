package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "helpers.go",
        "validate.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "validate_disabled.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "validate_disabled.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "validate_disabled.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "validate_disabled.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "validate_disabled.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "validate_disabled.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "validate_disabled.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "validate_disabled.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "validate_disabled.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "validate_disabled.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/security/apparmor",
    deps = [
        "//pkg/features:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//pkg/util/file:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/feature:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["validate_test.go"],
    data = [
        "testdata/profiles",
    ],
    embed = [":go_default_library"],
    deps = [
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
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
