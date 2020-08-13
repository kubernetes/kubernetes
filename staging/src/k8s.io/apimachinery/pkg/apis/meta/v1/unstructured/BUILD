package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "helpers_test.go",
        "unstructured_conversion_test.go",
        "unstructured_list_test.go",
        "unstructured_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/apitesting/fuzzer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/fuzzer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/testapigroup:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/testapigroup/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/testing:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/test:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/github.com/stretchr/testify/require:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "helpers.go",
        "unstructured.go",
        "unstructured_list.go",
        "zz_generated.deepcopy.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured",
    importpath = "k8s.io/apimachinery/pkg/apis/meta/v1/unstructured",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/json:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured/unstructuredscheme:all-srcs",
    ],
    tags = ["automanaged"],
)
