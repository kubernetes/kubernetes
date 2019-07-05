load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "install.go",
        "scheme.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubectl/scheme",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/admission/v1beta1:go_default_library",
        "//staging/src/k8s.io/api/admissionregistration/v1beta1:go_default_library",
        "//staging/src/k8s.io/api/apps/v1:go_default_library",
        "//staging/src/k8s.io/api/apps/v1beta1:go_default_library",
        "//staging/src/k8s.io/api/apps/v1beta2:go_default_library",
        "//staging/src/k8s.io/api/authentication/v1:go_default_library",
        "//staging/src/k8s.io/api/authentication/v1beta1:go_default_library",
        "//staging/src/k8s.io/api/authorization/v1:go_default_library",
        "//staging/src/k8s.io/api/authorization/v1beta1:go_default_library",
        "//staging/src/k8s.io/api/autoscaling/v1:go_default_library",
        "//staging/src/k8s.io/api/autoscaling/v2beta1:go_default_library",
        "//staging/src/k8s.io/api/batch/v1:go_default_library",
        "//staging/src/k8s.io/api/batch/v1beta1:go_default_library",
        "//staging/src/k8s.io/api/batch/v2alpha1:go_default_library",
        "//staging/src/k8s.io/api/certificates/v1beta1:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/extensions/v1beta1:go_default_library",
        "//staging/src/k8s.io/api/imagepolicy/v1alpha1:go_default_library",
        "//staging/src/k8s.io/api/networking/v1:go_default_library",
        "//staging/src/k8s.io/api/policy/v1beta1:go_default_library",
        "//staging/src/k8s.io/api/rbac/v1:go_default_library",
        "//staging/src/k8s.io/api/rbac/v1alpha1:go_default_library",
        "//staging/src/k8s.io/api/rbac/v1beta1:go_default_library",
        "//staging/src/k8s.io/api/scheduling/v1alpha1:go_default_library",
        "//staging/src/k8s.io/api/settings/v1alpha1:go_default_library",
        "//staging/src/k8s.io/api/storage/v1:go_default_library",
        "//staging/src/k8s.io/api/storage/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/scheme:go_default_library",
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
    srcs = ["sparse_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/api/batch/v1:go_default_library",
        "//staging/src/k8s.io/api/batch/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library",
    ],
)
