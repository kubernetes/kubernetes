package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_test",
)

go_test(
    name = "go_default_xtest",
    srcs = [
        "backward_compatibility_test.go",
        "conversion_test.go",
        "copy_test.go",
        "deep_copy_test.go",
        "serialization_test.go",
        "unstructured_test.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/testapi:go_default_library",
        "//pkg/api/testing:go_default_library",
        "//pkg/api/testing/compat:go_default_library",
        "//pkg/api/validation:go_default_library",
        "//vendor:github.com/golang/protobuf/proto",
        "//vendor:github.com/google/gofuzz",
        "//vendor:github.com/ugorji/go/codec",
        "//vendor:k8s.io/apimachinery/pkg/api/equality",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/api/testing",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/conversion/unstructured",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/runtime/serializer/streaming",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
        "//vendor:k8s.io/apimachinery/pkg/util/json",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/validation/field",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/apis/pkg/api",
        "//vendor:k8s.io/apis/pkg/api/v1",
        "//vendor:k8s.io/apis/pkg/apis/extensions",
        "//vendor:k8s.io/apis/pkg/apis/extensions/v1beta1",
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
