load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "dispatcher.go",
        "doc.go",
        "plugin.go",
        "reinvocationcontext.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/admission/plugin/webhook/mutating",
    importpath = "k8s.io/apiserver/pkg/admission/plugin/webhook/mutating",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/admission/v1:go_default_library",
        "//staging/src/k8s.io/api/admissionregistration/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer/json:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission/configuration:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission/metrics:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission/plugin/webhook:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission/plugin/webhook/errors:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission/plugin/webhook/generic:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission/plugin/webhook/request:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission/plugin/webhook/util:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/webhook:go_default_library",
        "//vendor/github.com/evanphx/json-patch:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
        "//vendor/k8s.io/utils/trace:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["plugin_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission/plugin/webhook/testing:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
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
