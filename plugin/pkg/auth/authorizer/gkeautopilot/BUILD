package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
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

go_library(
    name = "go_default_library",
    srcs = [
        "gkeautopilot.go",
        "config.go",
    ],
    importpath = "k8s.io/kubernetes/plugin/pkg/auth/authorizer/gkeautopilot",
    deps = [
        "//pkg/registry/rbac/validation:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//staging/src/k8s.io/client-go/listers/admissionregistration/v1:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/sigs.k8s.io/yaml:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "gkeautopilot_test.go",
        "config_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/registry/rbac/validation:go_default_library",
        "//staging/src/k8s.io/api/admissionregistration/v1:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//staging/src/k8s.io/client-go/listers/admissionregistration/v1:go_default_library",
        "//vendor/github.com/google/go-cmp/cmp:go_default_library",
    ],
)
