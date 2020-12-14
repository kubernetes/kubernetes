load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "nodeshutdown_manager_linux.go",
        "nodeshutdown_manager_others.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/nodeshutdown",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:aix": [
            "//pkg/kubelet/eviction:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:android": [
            "//pkg/features:go_default_library",
            "//pkg/kubelet/eviction:go_default_library",
            "//pkg/kubelet/nodeshutdown/systemd:go_default_library",
            "//pkg/kubelet/types:go_default_library",
            "//pkg/kubelet/util/format:go_default_library",
            "//staging/src/k8s.io/api/core/v1:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/util/clock:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
            "//vendor/github.com/godbus/dbus/v5:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//pkg/kubelet/eviction:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//pkg/kubelet/eviction:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//pkg/kubelet/eviction:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:illumos": [
            "//pkg/kubelet/eviction:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:ios": [
            "//pkg/kubelet/eviction:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:js": [
            "//pkg/kubelet/eviction:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//pkg/features:go_default_library",
            "//pkg/kubelet/eviction:go_default_library",
            "//pkg/kubelet/nodeshutdown/systemd:go_default_library",
            "//pkg/kubelet/types:go_default_library",
            "//pkg/kubelet/util/format:go_default_library",
            "//staging/src/k8s.io/api/core/v1:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/util/clock:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
            "//vendor/github.com/godbus/dbus/v5:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//pkg/kubelet/eviction:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//pkg/kubelet/eviction:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//pkg/kubelet/eviction:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//pkg/kubelet/eviction:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//pkg/kubelet/eviction:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//pkg/kubelet/eviction:go_default_library",
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
        "//pkg/kubelet/nodeshutdown/systemd:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["nodeshutdown_manager_linux_test.go"],
    embed = [":go_default_library"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//pkg/apis/scheduling:go_default_library",
            "//pkg/features:go_default_library",
            "//pkg/kubelet/nodeshutdown/systemd:go_default_library",
            "//staging/src/k8s.io/api/core/v1:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/util/clock:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
            "//staging/src/k8s.io/component-base/featuregate/testing:go_default_library",
            "//vendor/github.com/stretchr/testify/assert:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//pkg/apis/scheduling:go_default_library",
            "//pkg/features:go_default_library",
            "//pkg/kubelet/nodeshutdown/systemd:go_default_library",
            "//staging/src/k8s.io/api/core/v1:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
            "//staging/src/k8s.io/apimachinery/pkg/util/clock:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
            "//staging/src/k8s.io/component-base/featuregate/testing:go_default_library",
            "//vendor/github.com/stretchr/testify/assert:go_default_library",
        ],
        "//conditions:default": [],
    }),
)
