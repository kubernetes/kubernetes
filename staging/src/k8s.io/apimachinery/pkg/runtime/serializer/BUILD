package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "codec_test.go",
        "sparse_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/conversion:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer/testing:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//vendor/github.com/google/gofuzz:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/sigs.k8s.io/yaml:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "codec_factory.go",
        "negotiated_codec.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apimachinery/pkg/runtime/serializer",
    importpath = "k8s.io/apimachinery/pkg/runtime/serializer",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer/json:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer/protobuf:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer/recognizer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer/versioning:go_default_library",
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
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer/json:all-srcs",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer/protobuf:all-srcs",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer/recognizer:all-srcs",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer/streaming:all-srcs",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer/testing:all-srcs",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer/versioning:all-srcs",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer/yaml:all-srcs",
    ],
    tags = ["automanaged"],
)
