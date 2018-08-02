load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "defaults.go",
        "registrations.go",
        "scheme.go",
        "validation.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/apis/kubeadm/v1alpha3:go_default_library",
        "//cmd/kubeadm/app/constants:go_default_library",
        "//cmd/kubeadm/app/util:go_default_library",
        "//pkg/kubelet/apis/kubeletconfig:go_default_library",
        "//pkg/kubelet/apis/kubeletconfig/v1beta1:go_default_library",
        "//pkg/kubelet/apis/kubeletconfig/validation:go_default_library",
        "//pkg/proxy/apis/kubeproxyconfig:go_default_library",
        "//pkg/proxy/apis/kubeproxyconfig/v1alpha1:go_default_library",
        "//pkg/proxy/apis/kubeproxyconfig/validation:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//vendor/k8s.io/utils/pointer:go_default_library",
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
    srcs = ["validation_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//pkg/kubelet/apis/kubeletconfig:go_default_library",
        "//pkg/proxy/apis/kubeproxyconfig:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/utils/pointer:go_default_library",
    ],
)
