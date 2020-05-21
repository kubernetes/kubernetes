load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "accept.go",
        "rules.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/rules",
    importpath = "github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/rules",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/github.com/openshift/api/image/docker10:go_default_library",
        "//vendor/github.com/openshift/api/image/v1:go_default_library",
        "//vendor/github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/apis/imagepolicy/v1:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/image/imageutil:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/image/reference:go_default_library",
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
