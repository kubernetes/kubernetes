load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["authorizer.go"],
    importpath = "k8s.io/kubernetes/openshift-kube-apiserver/authorization/scopeauthorizer",
    visibility = ["//visibility:public"],
    deps = [
        "//plugin/pkg/auth/authorizer/rbac:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//staging/src/k8s.io/client-go/listers/rbac/v1:go_default_library",
        "//vendor/github.com/openshift/api/authorization/v1:go_default_library",
        "//vendor/github.com/openshift/apiserver-library-go/pkg/authorization/scope:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["authorizer_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//vendor/github.com/openshift/api/authorization/v1:go_default_library",
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
