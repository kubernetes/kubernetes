load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "builder.go",
        "claims.go",
        "doc.go",
        "errors.go",
        "jwt.go",
        "validation.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gopkg.in/square/go-jose.v2/jwt",
    importpath = "gopkg.in/square/go-jose.v2/jwt",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/gopkg.in/square/go-jose.v2:go_default_library",
        "//vendor/gopkg.in/square/go-jose.v2/json:go_default_library",
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
