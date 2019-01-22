package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "escalation_check.go",
        "helpers.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/registry/rbac",
    deps = [
        "//pkg/apis/rbac:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/conversion:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/endpoints/request:go_default_library",
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
        "//pkg/registry/rbac/clusterrole:all-srcs",
        "//pkg/registry/rbac/clusterrolebinding:all-srcs",
        "//pkg/registry/rbac/reconciliation:all-srcs",
        "//pkg/registry/rbac/rest:all-srcs",
        "//pkg/registry/rbac/role:all-srcs",
        "//pkg/registry/rbac/rolebinding:all-srcs",
        "//pkg/registry/rbac/validation:all-srcs",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = ["helpers_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/apis/core:go_default_library",
        "//pkg/apis/core/helper:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
    ],
)
