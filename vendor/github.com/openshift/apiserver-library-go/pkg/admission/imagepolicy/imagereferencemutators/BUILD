load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "interfaces.go",
        "meta.go",
        "pods.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/imagereferencemutators",
    importpath = "github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/imagereferencemutators",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/apis/apps:go_default_library",
        "//pkg/apis/batch:go_default_library",
        "//pkg/apis/core:go_default_library",
        "//staging/src/k8s.io/api/apps/v1:go_default_library",
        "//staging/src/k8s.io/api/apps/v1beta1:go_default_library",
        "//staging/src/k8s.io/api/apps/v1beta2:go_default_library",
        "//staging/src/k8s.io/api/batch/v1:go_default_library",
        "//staging/src/k8s.io/api/batch/v1beta1:go_default_library",
        "//staging/src/k8s.io/api/batch/v2alpha1:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/extensions/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//vendor/github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/apis/imagepolicy/v1:go_default_library",
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
