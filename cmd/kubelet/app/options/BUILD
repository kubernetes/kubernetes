package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "container_runtime.go",
        "globalflags.go",
        "globalflags_linux.go",
        "globalflags_other.go",
        "options.go",
        "osflags_others.go",
        "osflags_windows.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/kubelet/app/options",
    deps = [
        "//pkg/apis/core:go_default_library",
        "//pkg/credentialprovider/azure:go_default_library",
        "//pkg/credentialprovider/gcp:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/kubelet/apis:go_default_library",
        "//pkg/kubelet/apis/config:go_default_library",
        "//pkg/kubelet/apis/config/scheme:go_default_library",
        "//pkg/kubelet/apis/config/validation:go_default_library",
        "//pkg/kubelet/config:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//pkg/master/ports:go_default_library",
        "//pkg/util/flag:go_default_library",
        "//pkg/util/taints:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/component-base/cli/flag:go_default_library",
        "//staging/src/k8s.io/component-base/logs:go_default_library",
        "//staging/src/k8s.io/component-base/version/verflag:go_default_library",
        "//staging/src/k8s.io/kubelet/config/v1beta1:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/github.com/google/cadvisor/container/common:go_default_library",
            "//vendor/github.com/google/cadvisor/container/containerd:go_default_library",
            "//vendor/github.com/google/cadvisor/container/docker:go_default_library",
            "//vendor/github.com/google/cadvisor/container/raw:go_default_library",
            "//vendor/github.com/google/cadvisor/machine:go_default_library",
            "//vendor/github.com/google/cadvisor/manager:go_default_library",
            "//vendor/github.com/google/cadvisor/storage:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/google/cadvisor/container/common:go_default_library",
            "//vendor/github.com/google/cadvisor/container/containerd:go_default_library",
            "//vendor/github.com/google/cadvisor/container/docker:go_default_library",
            "//vendor/github.com/google/cadvisor/container/raw:go_default_library",
            "//vendor/github.com/google/cadvisor/machine:go_default_library",
            "//vendor/github.com/google/cadvisor/manager:go_default_library",
            "//vendor/github.com/google/cadvisor/storage:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = ["options_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//staging/src/k8s.io/component-base/cli/flag:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
    ],
)
