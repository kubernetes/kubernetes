licenses(["notice"])

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

filegroup(
    name = "testdata",
    srcs = [
        "testdata/v1/invalidPod.yaml",
        "testdata/v1/invalidPod1.json",
        "testdata/v1/invalidPod2.json",
        "testdata/v1/invalidPod3.json",
        "testdata/v1/invalidPod4.yaml",
        "testdata/v1/validPod.yaml",
    ],
    visibility = ["//build/visible_to:COMMON_testing"],
)

go_test(
    name = "go_default_test",
    srcs = ["schema_test.go"],
    data = [
        ":testdata",
        "//api/swagger-spec",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/testing:go_default_library",
        "//vendor/github.com/ghodss/yaml:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/api/extensions/v1beta1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/testing/fuzzer:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/yaml:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = ["schema.go"],
    tags = ["automanaged"],
    visibility = ["//build/visible_to:pkg_kubectl_validation_CONSUMERS"],
    deps = [
        "//pkg/api/util:go_default_library",
        "//vendor/github.com/emicklei/go-restful-swagger12:go_default_library",
        "//vendor/github.com/exponent-io/jsonpath:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/yaml:go_default_library",
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
