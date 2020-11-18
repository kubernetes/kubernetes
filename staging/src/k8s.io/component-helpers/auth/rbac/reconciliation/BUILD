package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "reconcile_role_test.go",
        "reconcile_rolebindings_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/api/rbac/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "clusterrole_interfaces.go",
        "clusterrolebinding_interfaces.go",
        "namespace.go",
        "reconcile_role.go",
        "reconcile_rolebindings.go",
        "role_interfaces.go",
        "rolebinding_interfaces.go",
        "zz_generated.deepcopy.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/component-helpers/auth/rbac/reconciliation",
    importpath = "k8s.io/component-helpers/auth/rbac/reconciliation",
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/rbac/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/typed/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/typed/rbac/v1:go_default_library",
        "//staging/src/k8s.io/component-helpers/auth/rbac/validation:go_default_library",
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
