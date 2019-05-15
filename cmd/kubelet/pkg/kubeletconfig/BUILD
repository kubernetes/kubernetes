package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "configsync.go",
        "controller.go",
        "watch.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/kubelet/pkg/kubeletconfig",
    deps = [
        "//cmd/kubelet/pkg/apis/config:go_default_library",
        "//cmd/kubelet/pkg/apis/config/validation:go_default_library",
        "//cmd/kubelet/pkg/kubeletconfig/checkpoint:go_default_library",
        "//cmd/kubelet/pkg/kubeletconfig/checkpoint/store:go_default_library",
        "//cmd/kubelet/pkg/kubeletconfig/status:go_default_library",
        "//cmd/kubelet/pkg/kubeletconfig/util/log:go_default_library",
        "//cmd/kubelet/pkg/kubeletconfig/util/panic:go_default_library",
        "//pkg/util/filesystem:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/typed/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//cmd/kubelet/pkg/kubeletconfig/checkpoint:all-srcs",
        "//cmd/kubelet/pkg/kubeletconfig/configfiles:all-srcs",
        "//cmd/kubelet/pkg/kubeletconfig/status:all-srcs",
        "//cmd/kubelet/pkg/kubeletconfig/util/codec:all-srcs",
        "//cmd/kubelet/pkg/kubeletconfig/util/equal:all-srcs",
        "//cmd/kubelet/pkg/kubeletconfig/util/files:all-srcs",
        "//cmd/kubelet/pkg/kubeletconfig/util/log:all-srcs",
        "//cmd/kubelet/pkg/kubeletconfig/util/panic:all-srcs",
        "//cmd/kubelet/pkg/kubeletconfig/util/test:all-srcs",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = ["controller_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//cmd/kubelet/pkg/kubeletconfig/checkpoint:go_default_library",
        "//cmd/kubelet/pkg/kubeletconfig/checkpoint/store:go_default_library",
        "//cmd/kubelet/pkg/kubeletconfig/status:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
    ],
)
