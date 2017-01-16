package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "events.go",
        "schema.go",
        "validation.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/pod:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/api/service:go_default_library",
        "//pkg/api/util:go_default_library",
        "//pkg/api/validation/genericvalidation:go_default_library",
        "//pkg/apis/storage/util:go_default_library",
        "//pkg/capabilities:go_default_library",
        "//pkg/security/apparmor:go_default_library",
        "//pkg/util/config:go_default_library",
        "//pkg/util/intstr:go_default_library",
        "//vendor:github.com/emicklei/go-restful/swagger",
        "//vendor:github.com/exponent-io/jsonpath",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1/unstructured",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1/validation",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/validation",
        "//vendor:k8s.io/apimachinery/pkg/util/validation/field",
        "//vendor:k8s.io/apimachinery/pkg/util/yaml",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "events_test.go",
        "schema_test.go",
        "validation_test.go",
    ],
    data = [
        "testdata/v1/invalidPod.yaml",
        "testdata/v1/invalidPod1.json",
        "testdata/v1/invalidPod2.json",
        "testdata/v1/invalidPod3.json",
        "testdata/v1/invalidPod4.yaml",
        "testdata/v1/validPod.yaml",
        "//api/swagger-spec",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/api/service:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/testing:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//pkg/apis/storage/util:go_default_library",
        "//pkg/capabilities:go_default_library",
        "//pkg/security/apparmor:go_default_library",
        "//pkg/util/intstr:go_default_library",
        "//vendor:github.com/ghodss/yaml",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/validation/field",
        "//vendor:k8s.io/apimachinery/pkg/util/yaml",
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
        "//pkg/api/validation/genericvalidation:all-srcs",
        "//pkg/api/validation/path:all-srcs",
    ],
    tags = ["automanaged"],
)
