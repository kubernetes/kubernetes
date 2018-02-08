load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "attributes.go",
        "conversion.go",
        "doc.go",
    ],
    importpath = "k8s.io/apiserver/pkg/admission/plugin/webhook/versioned",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["conversion_test.go"],
    embed = [":go_default_library"],
    importpath = "k8s.io/apiserver/pkg/admission/plugin/webhook/versioned",
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/apis/example:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/apis/example/v1:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/apis/example2/v1:go_default_library",
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
