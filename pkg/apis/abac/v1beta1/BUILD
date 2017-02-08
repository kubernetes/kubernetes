package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_test",
)
load("@io_kubernetes_build//defs:go.bzl", "api_group")

api_group(
    name = "go_default_library",
    srcs = [
        "conversion.go",
        "doc.go",
        "register.go",
        "types.go",
    ],
    gen_deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
    ],
    gen_types = [
        "conversion",
        "deepcopy",
        "defaulter",
    ],
    package = "pkg/apis/abac/v1beta1",
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/abac:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = ["conversion_test.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/abac:go_default_library",
        "//pkg/apis/abac/v1beta1:go_default_library",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
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
