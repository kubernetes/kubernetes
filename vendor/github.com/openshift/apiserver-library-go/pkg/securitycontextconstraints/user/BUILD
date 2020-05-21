load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "mustrunas.go",
        "mustrunasrange.go",
        "nonroot.go",
        "runasany.go",
        "types.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/user",
    importpath = "github.com/openshift/apiserver-library-go/pkg/securitycontextconstraints/user",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/apis/core:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//vendor/github.com/openshift/api/security/v1:go_default_library",
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
