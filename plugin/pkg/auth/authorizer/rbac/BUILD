package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "rbac.go",
        "subject_locator.go",
    ],
    deps = [
        "//pkg/apis/rbac:go_default_library",
        "//pkg/client/listers/rbac/internalversion:go_default_library",
        "//pkg/registry/rbac/validation:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "rbac_test.go",
        "subject_locator_test.go",
    ],
    library = ":go_default_library",
    deps = [
        "//pkg/apis/rbac:go_default_library",
        "//pkg/registry/rbac/validation:go_default_library",
        "//plugin/pkg/auth/authorizer/rbac/bootstrappolicy:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
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
        "//plugin/pkg/auth/authorizer/rbac/bootstrappolicy:all-srcs",
    ],
    tags = ["automanaged"],
)
