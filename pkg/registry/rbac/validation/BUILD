package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "policy_comparator_test.go",
        "rule_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/rbac:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "policy_comparator.go",
        "rule.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/rbac:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apiserver/pkg/authentication/serviceaccount",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
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
