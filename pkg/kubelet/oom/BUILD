load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "oom_watcher_linux.go",
        "oom_watcher_unsupported.go",
        "types.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/oom",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
            "//staging/src/k8s.io/client-go/tools/record:go_default_library",
            "//vendor/github.com/google/cadvisor/utils/oomparser:go_default_library",
            "//vendor/k8s.io/klog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:ios": [
            "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
            "//staging/src/k8s.io/client-go/tools/record:go_default_library",
            "//vendor/github.com/google/cadvisor/utils/oomparser:go_default_library",
            "//vendor/k8s.io/klog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = ["oom_watcher_linux_test.go"],
    embed = [":go_default_library"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//staging/src/k8s.io/api/core/v1:go_default_library",
            "//staging/src/k8s.io/client-go/tools/record:go_default_library",
            "//vendor/github.com/google/cadvisor/utils/oomparser:go_default_library",
            "//vendor/github.com/stretchr/testify/assert:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//staging/src/k8s.io/api/core/v1:go_default_library",
            "//staging/src/k8s.io/client-go/tools/record:go_default_library",
            "//vendor/github.com/google/cadvisor/utils/oomparser:go_default_library",
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
    visibility = ["//visibility:public"],
)
