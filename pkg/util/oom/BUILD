package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "oom.go",
        "oom_fake.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "oom_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "oom_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "oom_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "oom_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "oom_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "oom_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "oom_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "oom_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "oom_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "oom_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "oom_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/util/oom",
    deps = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//pkg/kubelet/cm/util:go_default_library",
            "//vendor/github.com/golang/glog:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "oom_linux_test.go",
        ],
        "//conditions:default": [],
    }),
    embed = [":go_default_library"],
    deps = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/stretchr/testify/assert:go_default_library",
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
