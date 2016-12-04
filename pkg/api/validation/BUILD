package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
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
        "//pkg/api/endpoints:go_default_library",
        "//pkg/api/meta:go_default_library",
        "//pkg/api/pod:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/api/service:go_default_library",
        "//pkg/api/util:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/apis/meta/v1/validation:go_default_library",
        "//pkg/apis/storage/util:go_default_library",
        "//pkg/capabilities:go_default_library",
        "//pkg/labels:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/security/apparmor:go_default_library",
        "//pkg/util/config:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//pkg/util/intstr:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/validation:go_default_library",
        "//pkg/util/validation/field:go_default_library",
        "//pkg/util/yaml:go_default_library",
        "//vendor:github.com/emicklei/go-restful/swagger",
        "//vendor:github.com/exponent-io/jsonpath",
        "//vendor:github.com/golang/glog",
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
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/api/service:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/testing:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/apis/storage/util:go_default_library",
        "//pkg/capabilities:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/security/apparmor:go_default_library",
        "//pkg/util/intstr:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/validation/field:go_default_library",
        "//pkg/util/yaml:go_default_library",
        "//vendor:github.com/ghodss/yaml",
    ],
)
