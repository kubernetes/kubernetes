load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "generated_expansion.go",
        "podsecuritypolicyreview.go",
        "podsecuritypolicyselfsubjectreview.go",
        "podsecuritypolicysubjectreview.go",
        "rangeallocation.go",
        "security_client.go",
        "securitycontextconstraints.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/client-go/security/clientset/versioned/typed/security/v1",
    importpath = "github.com/openshift/client-go/security/clientset/versioned/typed/security/v1",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//vendor/github.com/openshift/api/security/v1:go_default_library",
        "//vendor/github.com/openshift/client-go/security/clientset/versioned/scheme:go_default_library",
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
