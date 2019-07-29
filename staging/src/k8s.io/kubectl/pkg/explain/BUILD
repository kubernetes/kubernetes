load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "explain.go",
        "field_lookup.go",
        "fields_printer.go",
        "fields_printer_builder.go",
        "formatter.go",
        "model_printer.go",
        "recursive_fields_printer.go",
        "typename.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kubectl/pkg/explain",
    importpath = "k8s.io/kubectl/pkg/explain",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/util/proto:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "explain_test.go",
        "field_lookup_test.go",
        "fields_printer_test.go",
        "formatter_test.go",
        "model_printer_test.go",
        "recursive_fields_printer_test.go",
        "typename_test.go",
    ],
    data = [
        "test-recursive-swagger.json",
        "test-swagger.json",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/meta/testrestmapper:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/scheme:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/util/openapi/testing:go_default_library",
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
