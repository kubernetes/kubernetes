package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["abac.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/abac:go_default_library",
        "//pkg/apis/abac/latest:go_default_library",
        "//pkg/apis/abac/v0:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
    ],
)

filegroup(
    name = "example_policy",
    testonly = True,
    srcs = [
        "example_policy_file.jsonl",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["abac_test.go"],
    data = [
        ":example_policy",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/abac:go_default_library",
        "//pkg/apis/abac/v0:go_default_library",
        "//pkg/apis/abac/v1beta1:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
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
