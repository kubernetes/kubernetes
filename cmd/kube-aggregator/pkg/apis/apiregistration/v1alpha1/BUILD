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
        "register.go",
        "types.generated.go",
        "types.go",
    ],
    gen_deps = [
        "//pkg/api/v1:go_default_library",
        "//cmd/kube-aggregator/pkg/apis/apiregistration:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
    ],
    gen_types = [
        "conversion",
        "deepcopy",
    ],
    package = "cmd/kube-aggregator/pkg/apis/apiregistration/v1alpha1",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/ugorji/go/codec",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
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
