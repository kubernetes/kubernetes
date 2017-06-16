package(default_visibility = ["//visibility:public"])

load("//pkg/generated/openapi:def.bzl", "openapi_library")

licenses(["notice"])

openapi_library(
    name = "go_default_library",
    srcs = ["doc.go"],
    openapi_targets = [
        "cmd/libs/go2idl/client-gen/test_apis/testgroup/v1",
        "federation/apis/federation/v1beta1",
        "pkg/api/v1",
        "pkg/apis/abac/v0",
        "pkg/apis/abac/v1beta1",
        "pkg/apis/admission/v1alpha1",
        "pkg/apis/admissionregistration/v1alpha1",
        "pkg/apis/apps/v1beta1",
        "pkg/apis/authentication/v1",
        "pkg/apis/authentication/v1beta1",
        "pkg/apis/authorization/v1",
        "pkg/apis/authorization/v1beta1",
        "pkg/apis/autoscaling/v1",
        "pkg/apis/autoscaling/v2alpha1",
        "pkg/apis/batch/v1",
        "pkg/apis/batch/v2alpha1",
        "pkg/apis/certificates/v1beta1",
        "pkg/apis/componentconfig/v1alpha1",
        "pkg/apis/extensions/v1beta1",
        "pkg/apis/imagepolicy/v1alpha1",
        "pkg/apis/networking/v1",
        "pkg/apis/policy/v1beta1",
        "pkg/apis/rbac/v1alpha1",
        "pkg/apis/rbac/v1beta1",
        "pkg/apis/settings/v1alpha1",
        "pkg/apis/storage/v1",
        "pkg/apis/storage/v1beta1",
        "pkg/version",
    ],
    tags = ["automanaged"],
    vendor_targets = [
        "k8s.io/apimachinery/pkg/api/resource",
        "k8s.io/apimachinery/pkg/apis/meta/v1",
        "k8s.io/apimachinery/pkg/apis/meta/v1alpha1",
        "k8s.io/apimachinery/pkg/runtime",
        "k8s.io/apimachinery/pkg/util/intstr",
        "k8s.io/apimachinery/pkg/version",
        "k8s.io/apiserver/pkg/apis/audit/v1alpha1",
        "k8s.io/apiserver/pkg/apis/example/v1",
        "k8s.io/client-go/pkg/api/v1",
        "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1beta1",
        "k8s.io/metrics/pkg/apis/custom_metrics/v1alpha1",
        "k8s.io/metrics/pkg/apis/metrics/v1alpha1",
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
