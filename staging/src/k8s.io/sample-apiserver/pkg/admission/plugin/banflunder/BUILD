load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["admission.go"],
    importpath = "k8s.io/sample-apiserver/pkg/admission/plugin/banflunder",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/admission/wardleinitializer:go_default_library",
        "//pkg/apis/wardle:go_default_library",
        "//pkg/client/informers/internalversion:go_default_library",
        "//pkg/client/listers/wardle/internalversion:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission:go_default_library",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = ["admission_test.go"],
    importpath = "k8s.io/sample-apiserver/pkg/admission/plugin/banflunder_test",
    deps = [
        ":go_default_library",
        "//pkg/admission/wardleinitializer:go_default_library",
        "//pkg/apis/wardle:go_default_library",
        "//pkg/client/clientset/internalversion/fake:go_default_library",
        "//pkg/client/informers/internalversion:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission:go_default_library",
        "//vendor/k8s.io/client-go/testing:go_default_library",
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
