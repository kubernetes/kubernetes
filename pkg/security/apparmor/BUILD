package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "helpers.go",
        "validate.go",
        "validate_disabled.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/security/apparmor",
    deps = [
        "//pkg/api/v1/pod:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//vendor/k8s.io/utils/path:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["validate_test.go"],
    data = [
        "testdata/profiles",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
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
