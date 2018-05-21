package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "ipvs_test.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "ipvs_linux_test.go",
            "kernelcheck_linux_test.go",
        ],
        "//conditions:default": [],
    }),
    embed = [":go_default_library"],
    deps = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/docker/libnetwork/ipvs:go_default_library",
            "//vendor/k8s.io/utils/exec:go_default_library",
            "//vendor/k8s.io/utils/exec/testing:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_library(
    name = "go_default_library",
    srcs = [
        "ipvs.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "ipvs_unsupported.go",
            "kernelcheck_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "ipvs_unsupported.go",
            "kernelcheck_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "ipvs_unsupported.go",
            "kernelcheck_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "ipvs_unsupported.go",
            "kernelcheck_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "ipvs_linux.go",
            "kernelcheck_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "ipvs_unsupported.go",
            "kernelcheck_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "ipvs_unsupported.go",
            "kernelcheck_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "ipvs_unsupported.go",
            "kernelcheck_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "ipvs_unsupported.go",
            "kernelcheck_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "ipvs_unsupported.go",
            "kernelcheck_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "ipvs_unsupported.go",
            "kernelcheck_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/util/ipvs",
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/k8s.io/utils/exec:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/k8s.io/utils/exec:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/k8s.io/utils/exec:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/k8s.io/utils/exec:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/docker/libnetwork/ipvs:go_default_library",
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
            "//vendor/k8s.io/utils/exec:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//vendor/k8s.io/utils/exec:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/k8s.io/utils/exec:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/k8s.io/utils/exec:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/k8s.io/utils/exec:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/k8s.io/utils/exec:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/k8s.io/utils/exec:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//pkg/util/ipvs/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
