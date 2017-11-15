load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "builtin.go",
        "delegating.go",
    ],
    importpath = "k8s.io/apiserver/pkg/authorization/authorizerfactory",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/authentication/user:go_default_library",
        "//pkg/authorization/authorizer:go_default_library",
        "//plugin/pkg/authorizer/webhook:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/authorization/v1beta1:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["builtin_test.go"],
    importpath = "k8s.io/apiserver/pkg/authorization/authorizerfactory",
    library = ":go_default_library",
    deps = [
        "//pkg/authentication/user:go_default_library",
        "//pkg/authorization/authorizer:go_default_library",
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
