package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "checker_test.go",
        "reader_test.go",
    ],
    importpath = "k8s.io/apiserver/pkg/audit/policy",
    library = ":go_default_library",
    deps = [
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/github.com/stretchr/testify/require:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/apis/audit:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//vendor/k8s.io/apiserver/plugin/pkg/audit/webhook:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "checker.go",
        "reader.go",
    ],
    importpath = "k8s.io/apiserver/pkg/audit/policy",
    deps = [
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/apis/audit:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/apis/audit/v1alpha1:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/apis/audit/v1beta1:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/apis/audit/validation:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/audit:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
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
