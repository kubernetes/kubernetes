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
        "annotation_key_constants.go",
        "doc.go",
        "field_constants.go",
        "json.go",
        "objectreference.go",
        "register.go",
        "resource.go",
        "taint.go",
        "toleration.go",
        "types.go",
        "zz_generated.deepcopy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apimachinery/announced:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apimachinery/registered:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/conversion:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["taint_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
)

go_test(
    name = "go_default_xtest",
    srcs = [
        "conversion_test.go",
        "copy_test.go",
        "deep_copy_test.go",
        "defaulting_test.go",
        "meta_test.go",
        "serialization_proto_test.go",
        "serialization_test.go",
        "unstructured_test.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/testing:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/batch/v2alpha1:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//vendor/github.com/gogo/protobuf/proto:go_default_library",
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
        "//vendor/github.com/google/gofuzz:go_default_library",
        "//vendor/github.com/ugorji/go/codec:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/testing:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/conversion:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/conversion/unstructured:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer/protobuf:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer/streaming:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/json:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
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
        "//pkg/api/endpoints:all-srcs",
        "//pkg/api/errors:all-srcs",
        "//pkg/api/events:all-srcs",
        "//pkg/api/helper:all-srcs",
        "//pkg/api/install:all-srcs",
        "//pkg/api/meta:all-srcs",
        "//pkg/api/pod:all-srcs",
        "//pkg/api/ref:all-srcs",
        "//pkg/api/resource:all-srcs",
        "//pkg/api/service:all-srcs",
        "//pkg/api/testapi:all-srcs",
        "//pkg/api/testing:all-srcs",
        "//pkg/api/unversioned:all-srcs",
        "//pkg/api/util:all-srcs",
        "//pkg/api/v1:all-srcs",
        "//pkg/api/validation:all-srcs",
    ],
    tags = ["automanaged"],
)
