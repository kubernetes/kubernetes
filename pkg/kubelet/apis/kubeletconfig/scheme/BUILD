load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["scheme.go"],
    importpath = "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/scheme",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubelet/apis/kubeletconfig:go_default_library",
        "//pkg/kubelet/apis/kubeletconfig/v1alpha1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
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
    srcs = ["scheme_test.go"],
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/scheme",
    deps = [
        "//pkg/kubelet/apis/kubeletconfig/fuzzer:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/testing/roundtrip:go_default_library",
    ],
)
