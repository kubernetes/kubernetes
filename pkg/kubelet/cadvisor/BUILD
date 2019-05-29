package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "cadvisor_linux.go",
        "cadvisor_unsupported.go",
        "cadvisor_windows.go",
        "doc.go",
        "helpers_linux.go",
        "helpers_unsupported.go",
        "types.go",
        "util.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/cadvisor",
    deps = [
        "//pkg/apis/core/v1/helper:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/github.com/google/cadvisor/events:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v2:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/google/cadvisor/cache/memory:go_default_library",
            "//vendor/github.com/google/cadvisor/container:go_default_library",
            "//vendor/github.com/google/cadvisor/container/containerd/install:go_default_library",
            "//vendor/github.com/google/cadvisor/container/crio/install:go_default_library",
            "//vendor/github.com/google/cadvisor/container/docker/install:go_default_library",
            "//vendor/github.com/google/cadvisor/container/systemd/install:go_default_library",
            "//vendor/github.com/google/cadvisor/fs:go_default_library",
            "//vendor/github.com/google/cadvisor/manager:go_default_library",
            "//vendor/github.com/google/cadvisor/utils/cloudinfo/aws:go_default_library",
            "//vendor/github.com/google/cadvisor/utils/cloudinfo/azure:go_default_library",
            "//vendor/github.com/google/cadvisor/utils/cloudinfo/gce:go_default_library",
            "//vendor/github.com/google/cadvisor/utils/sysfs:go_default_library",
            "//vendor/k8s.io/klog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//pkg/kubelet/winstats:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = ["util_test.go"],
    embed = [":go_default_library"],
    deps = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//staging/src/k8s.io/api/core/v1:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
            "//vendor/github.com/google/cadvisor/container/crio:go_default_library",
            "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//pkg/kubelet/cadvisor/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
