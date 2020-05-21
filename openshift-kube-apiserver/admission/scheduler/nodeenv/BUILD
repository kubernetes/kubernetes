load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "admission.go",
        "intializers.go",
    ],
    importpath = "k8s.io/kubernetes/openshift-kube-apiserver/admission/scheduler/nodeenv",
    visibility = ["//visibility:public"],
    deps = [
        "//openshift-kube-apiserver/admission/scheduler/nodeenv/labelselector:go_default_library",
        "//pkg/apis/core:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission/initializer:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/listers/core/v1:go_default_library",
        "//vendor/github.com/openshift/api/project/v1:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["admission_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//openshift-kube-apiserver/admission/scheduler/nodeenv/labelselector:go_default_library",
        "//pkg/apis/core:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/client-go/listers/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/github.com/openshift/api/project/v1:go_default_library",
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
        "//openshift-kube-apiserver/admission/scheduler/nodeenv/labelselector:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
