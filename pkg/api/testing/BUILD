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
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//vendor:github.com/google/gofuzz",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/api/testing",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer",
        "//vendor:k8s.io/apimachinery/pkg/util/intstr",
        "//vendor:k8s.io/apis/pkg/api",
        "//vendor:k8s.io/apis/pkg/api/v1",
        "//vendor:k8s.io/apis/pkg/apis/autoscaling",
        "//vendor:k8s.io/apis/pkg/apis/batch",
        "//vendor:k8s.io/apis/pkg/apis/certificates",
        "//vendor:k8s.io/apis/pkg/apis/extensions",
        "//vendor:k8s.io/apis/pkg/apis/extensions/v1beta1",
        "//vendor:k8s.io/apis/pkg/apis/policy",
        "//vendor:k8s.io/apis/pkg/apis/rbac",
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
