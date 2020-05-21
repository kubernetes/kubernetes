load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["ingress_admission.go"],
    importpath = "k8s.io/kubernetes/openshift-kube-apiserver/admission/route",
    visibility = ["//visibility:public"],
    deps = [
        "//openshift-kube-apiserver/admission/route/apis/ingressadmission:go_default_library",
        "//openshift-kube-apiserver/admission/route/apis/ingressadmission/v1:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/networking:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission/initializer:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/config/helpers:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["ingress_admission_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//openshift-kube-apiserver/admission/route/apis/ingressadmission:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/networking:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
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
        "//openshift-kube-apiserver/admission/route/apis/ingressadmission:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
