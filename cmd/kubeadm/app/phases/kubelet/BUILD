load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "config.go",
        "flags.go",
        "kubelet.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/componentconfigs:go_default_library",
        "//cmd/kubeadm/app/constants:go_default_library",
        "//cmd/kubeadm/app/images:go_default_library",
        "//cmd/kubeadm/app/util:go_default_library",
        "//cmd/kubeadm/app/util/apiclient:go_default_library",
        "//cmd/kubeadm/app/util/initsystem:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/rbac/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/version:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "config_test.go",
        "flags_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/apis/kubeadm/v1beta2:go_default_library",
        "//cmd/kubeadm/app/constants:go_default_library",
        "//cmd/kubeadm/app/util/config:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/version:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/fake:go_default_library",
        "//staging/src/k8s.io/client-go/testing:go_default_library",
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
