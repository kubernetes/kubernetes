load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

exports_files(
    srcs = [
        "testdata/v1/validPod.yaml",
    ],
    visibility = ["//build/visible_to:COMMON_testing"],
)

go_test(
    name = "go_default_test",
    srcs = ["schema_test.go"],
    data = [
        ":testdata",
    ],
    embed = [":go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = ["schema.go"],
    importpath = "k8s.io/kubernetes/pkg/kubectl/validation",
    visibility = ["//build/visible_to:pkg_kubectl_validation_CONSUMERS"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/github.com/exponent-io/jsonpath:go_default_library",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//build/visible_to:pkg_kubectl_validation_CONSUMERS"],
)
