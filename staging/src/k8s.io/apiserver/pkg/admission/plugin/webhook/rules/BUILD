load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["rules.go"],
    importpath = "k8s.io/apiserver/pkg/admission/plugin/webhook/rules",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/k8s.io/api/admissionregistration/v1beta1:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["rules_test.go"],
    embed = [":go_default_library"],
    importpath = "k8s.io/apiserver/pkg/admission/plugin/webhook/rules",
    deps = [
        "//vendor/k8s.io/api/admissionregistration/v1beta1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/admission:go_default_library",
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
