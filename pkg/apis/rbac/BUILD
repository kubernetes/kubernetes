package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "helpers.go",
        "register.go",
        "types.go",
        "zz_generated.deepcopy.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/apis/rbac",
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/conversion:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
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
        "//pkg/apis/rbac/fuzzer:all-srcs",
        "//pkg/apis/rbac/install:all-srcs",
        "//pkg/apis/rbac/v1:all-srcs",
        "//pkg/apis/rbac/v1alpha1:all-srcs",
        "//pkg/apis/rbac/v1beta1:all-srcs",
        "//pkg/apis/rbac/validation:all-srcs",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_xtest",
    srcs = ["helpers_test.go"],
    importpath = "k8s.io/kubernetes/pkg/apis/rbac_test",
    deps = [
        ":go_default_library",
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/apis/rbac/install:go_default_library",
        "//pkg/apis/rbac/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/diff:go_default_library",
    ],
)
