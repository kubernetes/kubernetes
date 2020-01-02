load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["validators.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/cli-runtime/pkg/kustomize/k8sdeps/validator",
    importpath = "k8s.io/cli-runtime/pkg/kustomize/k8sdeps/validator",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/validation:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1/validation:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
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
