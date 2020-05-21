load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "validate_apiserver.go",
        "validation_wrapper.go",
    ],
    importpath = "k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/apiserver",
    visibility = ["//visibility:public"],
    deps = [
        "//openshift-kube-apiserver/admission/customresourcevalidation:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/validation:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//vendor/github.com/openshift/api/config/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/config/clientset/versioned/typed/config/v1:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/apiserver/admission/admissionrestconfig:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/crypto:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["validate_apiserver_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//vendor/github.com/openshift/api/config/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/config/clientset/versioned/fake:go_default_library",
        "//vendor/github.com/openshift/client-go/config/clientset/versioned/typed/config/v1:go_default_library",
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
