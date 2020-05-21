load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "defaulting_scc.go",
        "defaults.go",
        "validate_scc.go",
    ],
    importpath = "k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/securitycontextconstraints",
    visibility = ["//visibility:public"],
    deps = [
        "//openshift-kube-apiserver/admission/customresourcevalidation:go_default_library",
        "//openshift-kube-apiserver/admission/customresourcevalidation/securitycontextconstraints/validation:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//vendor/github.com/openshift/api/security/v1:go_default_library",
        "//vendor/github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/util:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["defaulting_scc_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//vendor/github.com/openshift/api/security/v1:go_default_library",
        "//vendor/github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/util:go_default_library",
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
        "//openshift-kube-apiserver/admission/customresourcevalidation/securitycontextconstraints/validation:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
