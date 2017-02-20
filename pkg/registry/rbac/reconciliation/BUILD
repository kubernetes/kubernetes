package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "reconcile_clusterrole_test.go",
        "reconcile_clusterrolebindings_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apis/pkg/api",
        "//vendor:k8s.io/apis/pkg/apis/rbac",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "reconcile_clusterrole.go",
        "reconcile_clusterrolebindings.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion:go_default_library",
        "//pkg/registry/rbac/validation:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apis/pkg/api",
        "//vendor:k8s.io/apis/pkg/apis/rbac",
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
