load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "config.go",
        "plugin.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/credentialprovider/plugin",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/credentialprovider:go_default_library",
        "//pkg/kubelet/apis/config:go_default_library",
        "//pkg/kubelet/apis/config/v1alpha1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer/json:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/kubelet/pkg/apis/credentialprovider:go_default_library",
        "//staging/src/k8s.io/kubelet/pkg/apis/credentialprovider/install:go_default_library",
        "//staging/src/k8s.io/kubelet/pkg/apis/credentialprovider/v1alpha1:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
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
        "config_test.go",
        "plugin_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/credentialprovider:go_default_library",
        "//pkg/kubelet/apis/config:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/kubelet/pkg/apis/credentialprovider:go_default_library",
        "//staging/src/k8s.io/kubelet/pkg/apis/credentialprovider/v1alpha1:go_default_library",
    ],
)
