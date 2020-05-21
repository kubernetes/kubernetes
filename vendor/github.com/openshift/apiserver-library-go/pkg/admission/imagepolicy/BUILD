load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "accept.go",
        "helpers.go",
        "imagepolicy.go",
        "intializers.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy",
    importpath = "github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/apis/core:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission/initializer:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/listers/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//vendor/github.com/hashicorp/golang-lru:go_default_library",
        "//vendor/github.com/openshift/api/image/v1:go_default_library",
        "//vendor/github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/apis/imagepolicy/v1:go_default_library",
        "//vendor/github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/apis/imagepolicy/validation:go_default_library",
        "//vendor/github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/imagereferencemutators:go_default_library",
        "//vendor/github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/rules:go_default_library",
        "//vendor/github.com/openshift/client-go/image/clientset/versioned:go_default_library",
        "//vendor/github.com/openshift/client-go/image/clientset/versioned/typed/image/v1:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/apiserver/admission/admissionrestconfig:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/apis/imagepolicy/v1:all-srcs",
        "//vendor/github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/apis/imagepolicy/validation:all-srcs",
        "//vendor/github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/imagereferencemutators:all-srcs",
        "//vendor/github.com/openshift/apiserver-library-go/pkg/admission/imagepolicy/rules:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
