package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "configsync.go",
        "controller.go",
        "rollback.go",
        "watch.go",
    ],
    deps = [
        "//pkg/kubelet/apis/kubeletconfig:go_default_library",
        "//pkg/kubelet/apis/kubeletconfig/validation:go_default_library",
        "//pkg/kubelet/kubeletconfig/checkpoint:go_default_library",
        "//pkg/kubelet/kubeletconfig/checkpoint/store:go_default_library",
        "//pkg/kubelet/kubeletconfig/configfiles:go_default_library",
        "//pkg/kubelet/kubeletconfig/status:go_default_library",
        "//pkg/kubelet/kubeletconfig/util/equal:go_default_library",
        "//pkg/kubelet/kubeletconfig/util/log:go_default_library",
        "//pkg/kubelet/kubeletconfig/util/panic:go_default_library",
        "//pkg/util/filesystem:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
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
        "//pkg/kubelet/kubeletconfig/checkpoint:all-srcs",
        "//pkg/kubelet/kubeletconfig/configfiles:all-srcs",
        "//pkg/kubelet/kubeletconfig/status:all-srcs",
        "//pkg/kubelet/kubeletconfig/util/codec:all-srcs",
        "//pkg/kubelet/kubeletconfig/util/equal:all-srcs",
        "//pkg/kubelet/kubeletconfig/util/files:all-srcs",
        "//pkg/kubelet/kubeletconfig/util/log:all-srcs",
        "//pkg/kubelet/kubeletconfig/util/panic:all-srcs",
        "//pkg/kubelet/kubeletconfig/util/test:all-srcs",
    ],
    tags = ["automanaged"],
)
