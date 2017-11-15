load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["basicauth.go"],
    importpath = "k8s.io/apiserver/plugin/pkg/authenticator/request/basicauth",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/authentication/authenticator:go_default_library",
        "//pkg/authentication/user:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["basicauth_test.go"],
    importpath = "k8s.io/apiserver/plugin/pkg/authenticator/request/basicauth",
    library = ":go_default_library",
    deps = [
        "//pkg/authentication/authenticator:go_default_library",
        "//pkg/authentication/user:go_default_library",
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
