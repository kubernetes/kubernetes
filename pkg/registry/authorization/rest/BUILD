package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["storage_authorization.go"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apis/authorization:go_default_library",
        "//pkg/registry/authorization/localsubjectaccessreview:go_default_library",
        "//pkg/registry/authorization/selfsubjectaccessreview:go_default_library",
        "//pkg/registry/authorization/subjectaccessreview:go_default_library",
        "//vendor/k8s.io/api/authorization/v1:go_default_library",
        "//vendor/k8s.io/api/authorization/v1beta1:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/registry/generic:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/registry/rest:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/server:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/server/storage:go_default_library",
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
