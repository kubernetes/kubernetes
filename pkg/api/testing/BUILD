package(default_visibility = ["//visibility:public"])

licenses(["notice"])

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
    tags = ["automanaged"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm/fuzzer:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/admissionregistration:go_default_library",
        "//pkg/apis/apps:go_default_library",
        "//pkg/apis/autoscaling:go_default_library",
        "//pkg/apis/batch:go_default_library",
        "//pkg/apis/certificates:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//pkg/apis/policy:go_default_library",
        "//pkg/apis/rbac:go_default_library",
        "//vendor/github.com/google/gofuzz:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/testing:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
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
