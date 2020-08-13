package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["builtin_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "builtin.go",
        "delegating.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/authorization/authorizerfactory",
    importpath = "k8s.io/apiserver/pkg/authorization/authorizerfactory",
    deps = [
        "//staging/src/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//staging/src/k8s.io/apiserver/plugin/pkg/authorizer/webhook:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/typed/authorization/v1:go_default_library",
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
)
