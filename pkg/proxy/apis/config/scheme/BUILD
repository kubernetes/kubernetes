load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["scheme.go"],
    importpath = "k8s.io/kubernetes/pkg/proxy/apis/config/scheme",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/proxy/apis/config:go_default_library",
        "//pkg/proxy/apis/config/v1alpha1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
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
    srcs = ["scheme_test.go"],
    data = glob(["testdata/**"]),
    embed = [":go_default_library"],
    deps = [
        "//pkg/proxy/apis/config/fuzzer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/apitesting/roundtrip:go_default_library",
        "//staging/src/k8s.io/component-base/config/testing:go_default_library",
    ],
)
