package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "eviction_manager_test.go",
        "helpers_test.go",
        "memory_threshold_notifier_test.go",
        "mock_threshold_notifier_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/apis/core:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/kubelet/apis/stats/v1alpha1:go_default_library",
        "//pkg/kubelet/eviction/api:go_default_library",
        "//pkg/kubelet/lifecycle:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/clock:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        "//staging/src/k8s.io/component-base/featuregate/testing:go_default_library",
        "//vendor/github.com/stretchr/testify/mock:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "eviction_manager.go",
        "helpers.go",
        "memory_threshold_notifier.go",
        "threshold_notifier_linux.go",
        "threshold_notifier_unsupported.go",
        "types.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/eviction",
    deps = [
        "//pkg/api/v1/resource:go_default_library",
        "//pkg/apis/core/v1/helper:go_default_library",
        "//pkg/apis/core/v1/helper/qos:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/kubelet/apis/stats/v1alpha1:go_default_library",
        "//pkg/kubelet/cm:go_default_library",
        "//pkg/kubelet/eviction/api:go_default_library",
        "//pkg/kubelet/lifecycle:go_default_library",
        "//pkg/kubelet/metrics:go_default_library",
        "//pkg/kubelet/pod:go_default_library",
        "//pkg/kubelet/server/stats:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//pkg/kubelet/util/format:go_default_library",
        "//pkg/scheduler/api:go_default_library",
        "//pkg/scheduler/util:go_default_library",
        "//pkg/volume/util:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/clock:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
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
        "//pkg/kubelet/eviction/api:all-srcs",
    ],
    tags = ["automanaged"],
)
