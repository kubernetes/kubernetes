load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["openidmetadata.go"],
    importpath = "k8s.io/kubernetes/test/images/agnhost/openidmetadata",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//vendor/github.com/coreos/go-oidc:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/golang.org/x/oauth2:go_default_library",
        "//vendor/gopkg.in/square/go-jose.v2/jwt:go_default_library",
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
