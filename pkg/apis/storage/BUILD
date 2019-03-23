package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "register.go",
        "types.go",
        "zz_generated.deepcopy.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/apis/storage",
    deps = [
        "//pkg/apis/core:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
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
        "//pkg/apis/storage/fuzzer:all-srcs",
        "//pkg/apis/storage/install:all-srcs",
        "//pkg/apis/storage/util:all-srcs",
        "//pkg/apis/storage/v1:all-srcs",
        "//pkg/apis/storage/v1alpha1:all-srcs",
        "//pkg/apis/storage/v1beta1:all-srcs",
        "//pkg/apis/storage/validation:all-srcs",
    ],
    tags = ["automanaged"],
)
