package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["admission.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/sample-apiserver/pkg/admission/plugin/banflunder",
    importpath = "k8s.io/sample-apiserver/pkg/admission/plugin/banflunder",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/admission/wardleinitializer:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/apis/wardle:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/generated/informers/externalversions:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/generated/listers/wardle/v1alpha1:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["admission_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/client-go/testing:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/admission/wardleinitializer:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/apis/wardle/v1alpha1:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/generated/clientset/versioned/fake:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/generated/informers/externalversions:go_default_library",
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
)
