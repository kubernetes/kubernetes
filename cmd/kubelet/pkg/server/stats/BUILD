load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "fs_resource_analyzer.go",
        "handler.go",
        "prometheus_resource_metrics.go",
        "resource_analyzer.go",
        "summary.go",
        "summary_sys_containers.go",
        "summary_sys_containers_windows.go",
        "volume_stat_calculator.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/kubelet/pkg/server/stats",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/kubelet/pkg/apis/stats/v1alpha1:go_default_library",
        "//cmd/kubelet/pkg/cm:go_default_library",
        "//cmd/kubelet/pkg/container:go_default_library",
        "//cmd/kubelet/pkg/util:go_default_library",
        "//cmd/kubelet/pkg/util/format:go_default_library",
        "//pkg/volume:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/github.com/emicklei/go-restful:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "prometheus_resource_metrics_test.go",
        "summary_test.go",
        "summary_windows_test.go",
        "volume_stat_calculator_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//cmd/kubelet/pkg/apis/stats/v1alpha1:go_default_library",
        "//cmd/kubelet/pkg/server/stats/testing:go_default_library",
        "//pkg/volume:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/github.com/prometheus/client_model/go:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/github.com/stretchr/testify/mock:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//cmd/kubelet/pkg/cm:go_default_library",
            "//vendor/github.com/google/gofuzz:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//cmd/kubelet/pkg/cm:go_default_library",
            "//vendor/github.com/google/gofuzz:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//cmd/kubelet/pkg/cm:go_default_library",
            "//vendor/github.com/google/gofuzz:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//cmd/kubelet/pkg/cm:go_default_library",
            "//vendor/github.com/google/gofuzz:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//cmd/kubelet/pkg/cm:go_default_library",
            "//vendor/github.com/google/gofuzz:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//cmd/kubelet/pkg/cm:go_default_library",
            "//vendor/github.com/google/gofuzz:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//cmd/kubelet/pkg/cm:go_default_library",
            "//vendor/github.com/google/gofuzz:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//cmd/kubelet/pkg/cm:go_default_library",
            "//vendor/github.com/google/gofuzz:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//cmd/kubelet/pkg/cm:go_default_library",
            "//vendor/github.com/google/gofuzz:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//cmd/kubelet/pkg/cm:go_default_library",
            "//vendor/github.com/google/gofuzz:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//cmd/kubelet/pkg/cm:go_default_library",
            "//vendor/github.com/google/gofuzz:go_default_library",
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
        "//cmd/kubelet/pkg/server/stats/testing:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
