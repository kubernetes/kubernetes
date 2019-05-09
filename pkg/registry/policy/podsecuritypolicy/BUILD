package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "strategy.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/registry/policy/podsecuritypolicy",
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/api/podsecuritypolicy:go_default_library",
        "//pkg/apis/policy:go_default_library",
        "//pkg/apis/policy/validation:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/registry/rest:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/names:go_default_library",
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
        "//pkg/registry/policy/podsecuritypolicy/storage:all-srcs",
    ],
    tags = ["automanaged"],
)
