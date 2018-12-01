load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["strict.go"],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/util/config/strict",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm/scheme:go_default_library",
        "//cmd/kubeadm/app/componentconfigs:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
        "//vendor/sigs.k8s.io/yaml:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["strict_test.go"],
    data = glob(["testdata/**"]),
    embed = [":go_default_library"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm/v1alpha3:go_default_library",
        "//cmd/kubeadm/app/apis/kubeadm/v1beta1:go_default_library",
        "//cmd/kubeadm/app/componentconfigs:go_default_library",
        "//cmd/kubeadm/app/constants:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/kube-proxy/config/v1alpha1:go_default_library",
        "//staging/src/k8s.io/kubelet/config/v1beta1:go_default_library",
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
