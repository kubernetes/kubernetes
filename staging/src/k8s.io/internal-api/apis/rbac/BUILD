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
    importmap = "k8s.io/kubernetes/vendor/k8s.io/internal-api/apis/rbac",
    importpath = "k8s.io/internal-api/apis/rbac",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
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
        "//staging/src/k8s.io/internal-api/apis/rbac/fuzzer:all-srcs",
        "//staging/src/k8s.io/internal-api/apis/rbac/install:all-srcs",
        "//staging/src/k8s.io/internal-api/apis/rbac/v1:all-srcs",
        "//staging/src/k8s.io/internal-api/apis/rbac/v1alpha1:all-srcs",
        "//staging/src/k8s.io/internal-api/apis/rbac/v1beta1:all-srcs",
        "//staging/src/k8s.io/internal-api/apis/rbac/validation:all-srcs",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = ["helpers_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//staging/src/k8s.io/internal-api/api/legacyscheme:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/rbac/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/rbac/v1:go_default_library",
    ],
)
