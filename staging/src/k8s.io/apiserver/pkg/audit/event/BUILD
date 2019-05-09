load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["attributes.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/audit/event",
    importpath = "k8s.io/apiserver/pkg/audit/event",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apiserver/pkg/apis/audit:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = ["attributes_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apiserver/pkg/apis/audit:go_default_library",
        "//vendor/github.com/stretchr/testify/require:go_default_library",
    ],
)
