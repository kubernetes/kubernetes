package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load("@io_kubernetes_build//defs:go.bzl", "api_group")

api_group(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "register.go",
        "types.go",
    ],
    gen_deps = ["//vendor:k8s.io/apimachinery/pkg/conversion"],
    gen_types = ["deepcopy"],
    package = "pkg/apis/authentication",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
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
        "//pkg/apis/authentication/install:all-srcs",
        "//pkg/apis/authentication/v1beta1:all-srcs",
    ],
    tags = ["automanaged"],
)
