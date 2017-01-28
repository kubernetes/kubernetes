package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load("@io_kubernetes_build//defs:go.bzl", "api_group")

api_group(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "helpers.go",
        "register.go",
        "types.go",
    ],
    gen_deps = ["//vendor:k8s.io/apimachinery/pkg/conversion"],
    gen_types = ["deepcopy"],
    package = "pkg/apis/rbac",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
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
        "//pkg/apis/rbac/install:all-srcs",
        "//pkg/apis/rbac/v1alpha1:all-srcs",
        "//pkg/apis/rbac/v1beta1:all-srcs",
        "//pkg/apis/rbac/validation:all-srcs",
    ],
    tags = ["automanaged"],
)
