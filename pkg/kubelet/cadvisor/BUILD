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
        "types.go",
        "util.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "cadvisor_unsupported.go",
            "helpers_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "cadvisor_unsupported.go",
            "helpers_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "cadvisor_unsupported.go",
            "helpers_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "cadvisor_unsupported.go",
            "helpers_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "cadvisor_linux.go",
            "cadvisor_unsupported.go",
            "helpers_linux.go",
            "helpers_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "cadvisor_unsupported.go",
            "helpers_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "cadvisor_unsupported.go",
            "helpers_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "cadvisor_unsupported.go",
            "helpers_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "cadvisor_unsupported.go",
            "helpers_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "cadvisor_unsupported.go",
            "helpers_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "cadvisor_windows.go",
            "helpers_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/kubelet/cadvisor",
    deps = [
        "//pkg/apis/core/v1/helper:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//vendor/github.com/google/cadvisor/events:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v2:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/feature:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/golang/glog:go_default_library",
            "//vendor/github.com/google/cadvisor/cache/memory:go_default_library",
            "//vendor/github.com/google/cadvisor/container:go_default_library",
            "//vendor/github.com/google/cadvisor/fs:go_default_library",
            "//vendor/github.com/google/cadvisor/http:go_default_library",
            "//vendor/github.com/google/cadvisor/manager:go_default_library",
            "//vendor/github.com/google/cadvisor/metrics:go_default_library",
            "//vendor/github.com/google/cadvisor/utils/sysfs:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//pkg/kubelet/winstats:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "cadvisor_linux_test.go",
            "util_test.go",
        ],
        "//conditions:default": [],
    }),
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/pkg/kubelet/cadvisor",
    deps = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//pkg/apis/core/v1/helper:go_default_library",
            "//pkg/features:go_default_library",
            "//pkg/kubelet/types:go_default_library",
            "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
            "//vendor/github.com/google/cadvisor/metrics:go_default_library",
            "//vendor/github.com/stretchr/testify/assert:go_default_library",
            "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
            "//vendor/k8s.io/apiserver/pkg/util/feature:go_default_library",
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
        "//pkg/kubelet/cadvisor/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
