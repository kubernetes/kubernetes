package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "conversion.go",
        "fuzzer.go",
        "pod_specs.go",
    ],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm/fuzzer:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/fuzzer:go_default_library",
        "//pkg/apis/admissionregistration/fuzzer:go_default_library",
        "//pkg/apis/apps/fuzzer:go_default_library",
        "//pkg/apis/autoscaling/fuzzer:go_default_library",
        "//pkg/apis/batch/fuzzer:go_default_library",
        "//pkg/apis/certificates/fuzzer:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/extensions/fuzzer:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//pkg/apis/networking/fuzzer:go_default_library",
        "//pkg/apis/policy/fuzzer:go_default_library",
        "//pkg/apis/rbac/fuzzer:go_default_library",
        "//pkg/apis/storage/fuzzer:go_default_library",
        "//vendor/github.com/google/gofuzz:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/testing:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/testing/fuzzer:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/fuzzer:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/fields:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//pkg/api/testing/compat:all-srcs",
    ],
    tags = ["automanaged"],
)
