load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bypriority.go",
        "byrestrictions.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/util/sort",
    importpath = "github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/util/sort",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//vendor/github.com/openshift/api/security/v1:go_default_library",
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
