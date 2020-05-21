load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["converter.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/apiserver-library-go/pkg/authorization/scope",
    importpath = "github.com/openshift/apiserver-library-go/pkg/authorization/scope",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/apis/rbac/v1:go_default_library",
        "//plugin/pkg/auth/authorizer/rbac:go_default_library",
        "//staging/src/k8s.io/api/rbac/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//staging/src/k8s.io/client-go/listers/rbac/v1:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/authorization/scopemetadata:go_default_library",
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
