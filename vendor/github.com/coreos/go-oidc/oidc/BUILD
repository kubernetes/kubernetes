load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "client.go",
        "doc.go",
        "identity.go",
        "interface.go",
        "key.go",
        "provider.go",
        "transport.go",
        "util.go",
        "verification.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/coreos/go-oidc/http:go_default_library",
        "//vendor/github.com/coreos/go-oidc/jose:go_default_library",
        "//vendor/github.com/coreos/go-oidc/key:go_default_library",
        "//vendor/github.com/coreos/go-oidc/oauth2:go_default_library",
        "//vendor/github.com/coreos/pkg/timeutil:go_default_library",
        "//vendor/github.com/jonboulle/clockwork:go_default_library",
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
