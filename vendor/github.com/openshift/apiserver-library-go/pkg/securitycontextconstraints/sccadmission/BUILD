load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "admission.go",
        "intializers.go",
        "scc_exec.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/sccadmission",
    importpath = "github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/sccadmission",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/apis/core:go_default_library",
        "//pkg/apis/core/helper:go_default_library",
        "//pkg/apis/core/v1:go_default_library",
        "//pkg/registry/rbac:go_default_library",
        "//pkg/serviceaccount:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission/initializer:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/github.com/openshift/api/security/v1:go_default_library",
        "//vendor/github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/sccmatching:go_default_library",
        "//vendor/github.com/openshift/client-go/security/informers/externalversions/security/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/security/listers/security/v1:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
