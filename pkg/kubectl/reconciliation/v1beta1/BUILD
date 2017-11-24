load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "clusterrole_interfaces.go",
        "clusterrolebinding_interfaces.go",
        "reconcile_role.go",
        "reconcile_rolebindings.go",
        "role_interfaces.go",
        "rolebinding_interfaces.go",
        "zz_generated.deepcopy.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubectl/reconciliation/v1beta1",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/api/rbac/v1beta1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/core/v1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/rbac/v1beta1:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "reconcile_role_test.go",
        "reconcile_rolebindings_test.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubectl/reconciliation/v1beta1",
    library = ":go_default_library",
    deps = [
        "//pkg/apis/core/helper:go_default_library",
        "//vendor/k8s.io/api/rbac/v1beta1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/diff:go_default_library",
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
