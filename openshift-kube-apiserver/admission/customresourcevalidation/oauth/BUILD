load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "helpers.go",
        "validate_github.go",
        "validate_gitlab.go",
        "validate_google.go",
        "validate_idp.go",
        "validate_keystone.go",
        "validate_ldap.go",
        "validate_oauth.go",
        "validate_openid.go",
        "validate_requestheader.go",
    ],
    importpath = "k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/oauth",
    visibility = ["//visibility:public"],
    deps = [
        "//openshift-kube-apiserver/admission/customresourcevalidation:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/validation:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/validation/path:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//vendor/github.com/openshift/api/config/v1:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/config/validation:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/security/ldaputil:go_default_library",
        "//vendor/k8s.io/utils/pointer:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "validate_github_test.go",
        "validate_gitlab_test.go",
        "validate_google_test.go",
        "validate_idp_test.go",
        "validate_keystone_test.go",
        "validate_ldap_test.go",
        "validate_openid_test.go",
        "validate_requestheader_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//vendor/github.com/openshift/api/config/v1:go_default_library",
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
