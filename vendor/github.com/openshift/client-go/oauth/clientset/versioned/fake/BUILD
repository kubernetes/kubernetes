load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "clientset_generated.go",
        "doc.go",
        "register.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/client-go/oauth/clientset/versioned/fake",
    importpath = "github.com/openshift/client-go/oauth/clientset/versioned/fake",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//staging/src/k8s.io/client-go/discovery:go_default_library",
        "//staging/src/k8s.io/client-go/discovery/fake:go_default_library",
        "//staging/src/k8s.io/client-go/testing:go_default_library",
        "//vendor/github.com/openshift/api/oauth/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/oauth/clientset/versioned:go_default_library",
        "//vendor/github.com/openshift/client-go/oauth/clientset/versioned/typed/oauth/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/oauth/clientset/versioned/typed/oauth/v1/fake:go_default_library",
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
