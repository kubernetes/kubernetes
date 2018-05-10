load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "cadvisor_stats_provider.go",
        "cri_stats_provider.go",
        "helper.go",
        "log_metrics_provider.go",
        "stats_provider.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "stats_provider_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "stats_provider_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "stats_provider_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "stats_provider_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "stats_provider_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "stats_provider_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "stats_provider_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "stats_provider_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "stats_provider_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "stats_provider_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "stats_provider_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/kubelet/stats",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubelet/apis/cri:go_default_library",
        "//pkg/kubelet/apis/cri/runtime/v1alpha3:go_default_library",
        "//pkg/kubelet/apis/stats/v1alpha1:go_default_library",
        "//pkg/kubelet/cadvisor:go_default_library",
        "//pkg/kubelet/cm:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/kuberuntime:go_default_library",
        "//pkg/kubelet/leaky:go_default_library",
        "//pkg/kubelet/pod:go_default_library",
        "//pkg/kubelet/server/stats:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//pkg/volume:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/google/cadvisor/fs:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v2:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
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
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "cadvisor_stats_provider_test.go",
        "cri_stats_provider_test.go",
        "helper_test.go",
        "log_metrics_provider_test.go",
        "stats_provider_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/kubelet/apis/cri/runtime/v1alpha3:go_default_library",
        "//pkg/kubelet/apis/cri/testing:go_default_library",
        "//pkg/kubelet/apis/stats/v1alpha1:go_default_library",
        "//pkg/kubelet/cadvisor/testing:go_default_library",
        "//pkg/kubelet/cm:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/container/testing:go_default_library",
        "//pkg/kubelet/kuberuntime:go_default_library",
        "//pkg/kubelet/leaky:go_default_library",
        "//pkg/kubelet/pod/testing:go_default_library",
        "//pkg/kubelet/server/stats:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//pkg/volume:go_default_library",
        "//vendor/github.com/google/cadvisor/fs:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v2:go_default_library",
        "//vendor/github.com/google/gofuzz:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/github.com/stretchr/testify/require:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
    ],
)
