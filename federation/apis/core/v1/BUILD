package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_kubernetes_build//defs:go.bzl",
    "api_group",
)

api_group(
    name = "go_default_library",
    srcs = [
        "conversion.go",
        "defaults.go",
        "doc.go",
        "register.go",
    ],
    gen_deps = ["//federation/apis/core:go_default_library"],
    gen_types = ["conversion"],
    package = "federation/apis/core/v1",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
