load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["kubeconfig.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/admission/plugin/webhook/config",
    importpath = "k8s.io/apiserver/pkg/admission/plugin/webhook/config",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission/plugin/webhook/config/apis/webhookadmission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission/plugin/webhook/config/apis/webhookadmission/v1:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission/plugin/webhook/config/apis/webhookadmission/v1alpha1:go_default_library",
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
        "//staging/src/k8s.io/apiserver/pkg/admission/plugin/webhook/config/apis/webhookadmission:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["kubeconfig_test.go"],
    embed = [":go_default_library"],
)
