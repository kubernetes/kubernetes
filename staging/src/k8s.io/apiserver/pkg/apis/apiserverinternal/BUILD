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
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/apis/apiserverinternal",
    importpath = "k8s.io/apiserver/pkg/apis/apiserverinternal",
    deps = [
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
        "//staging/src/k8s.io/apiserver/pkg/apis/apiserverinternal/fuzzer:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/apis/apiserverinternal/install:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/apis/apiserverinternal/v1alpha1:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/apis/apiserverinternal/validation:all-srcs",
    ],
    tags = ["automanaged"],
)
