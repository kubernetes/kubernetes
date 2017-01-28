package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_kubernetes_build//defs:go.bzl",
    "api_group",
)

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
    package = "cmd/kube-aggregator/pkg/apis/apiregistration",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
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
        "//cmd/kube-aggregator/pkg/apis/apiregistration/install:all-srcs",
        "//cmd/kube-aggregator/pkg/apis/apiregistration/v1alpha1:all-srcs",
        "//cmd/kube-aggregator/pkg/apis/apiregistration/validation:all-srcs",
    ],
    tags = ["automanaged"],
)
