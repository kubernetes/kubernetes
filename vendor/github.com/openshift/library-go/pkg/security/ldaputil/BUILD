load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "attribute.go",
        "url.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/library-go/pkg/security/ldaputil",
    importpath = "github.com/openshift/library-go/pkg/security/ldaputil",
    visibility = ["//visibility:public"],
    deps = ["//vendor/gopkg.in/ldap.v2:go_default_library"],
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
