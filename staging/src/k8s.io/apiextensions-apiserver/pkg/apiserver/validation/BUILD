package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["validation.go"],
    importpath = "k8s.io/apiextensions-apiserver/pkg/apiserver/validation",
    deps = [
        "//vendor/github.com/go-openapi/spec:go_default_library",
        "//vendor/github.com/go-openapi/strfmt:go_default_library",
        "//vendor/github.com/go-openapi/validate:go_default_library",
        "//vendor/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = ["validation_test.go"],
    importpath = "k8s.io/apiextensions-apiserver/pkg/apiserver/validation",
    library = ":go_default_library",
    deps = [
        "//vendor/github.com/go-openapi/spec:go_default_library",
        "//vendor/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions:go_default_library",
        "//vendor/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/fuzzer:go_default_library",
        "//vendor/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/testing/fuzzer:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/json:go_default_library",
    ],
)
