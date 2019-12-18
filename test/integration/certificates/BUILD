load("@io_bazel_rules_go//go:def.bzl", "go_test")

go_test(
    name = "go_default_test",
    srcs = [
        "defaulting_test.go",
        "field_selector_test.go",
        "main_test.go",
    ],
    tags = ["integration"],
    deps = [
        "//staging/src/k8s.io/api/certificates/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/typed/certificates/v1beta1:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//test/integration/framework:go_default_library",
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
