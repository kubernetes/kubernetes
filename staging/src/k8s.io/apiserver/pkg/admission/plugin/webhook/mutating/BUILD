load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "admission.go",
        "doc.go",
    ],
    importpath = "k8s.io/apiserver/pkg/admission/plugin/webhook/mutating",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/evanphx/json-patch:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/api/admission/v1beta1:go_default_library",
        "//vendor/k8s.io/api/admissionregistration/v1beta1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer/json:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission/configuration:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission/initializer:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission/metrics:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission/plugin/webhook/config:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission/plugin/webhook/errors:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission/plugin/webhook/namespace:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission/plugin/webhook/request:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission/plugin/webhook/rules:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission/plugin/webhook/versioned:go_default_library",
        "//vendor/k8s.io/client-go/informers:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["admission_test.go"],
    embed = [":go_default_library"],
    importpath = "k8s.io/apiserver/pkg/admission/plugin/webhook/mutating",
    deps = [
        "//vendor/k8s.io/api/admission/v1beta1:go_default_library",
        "//vendor/k8s.io/api/admissionregistration/v1beta1:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission/plugin/webhook/config:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission/plugin/webhook/testcerts:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
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
