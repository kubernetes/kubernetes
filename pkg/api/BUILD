load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

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
    importpath = "k8s.io/kubernetes/pkg/api",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/internalversion:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/conversion:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["taint_test.go"],
    importpath = "k8s.io/kubernetes/pkg/api",
    library = ":go_default_library",
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
        "//pkg/api/events:all-srcs",
        "//pkg/api/fuzzer:all-srcs",
        "//pkg/api/helper:all-srcs",
        "//pkg/api/install:all-srcs",
        "//pkg/api/legacyscheme:all-srcs",
        "//pkg/api/persistentvolume:all-srcs",
        "//pkg/api/pod:all-srcs",
        "//pkg/api/ref:all-srcs",
        "//pkg/api/resource:all-srcs",
        "//pkg/api/service:all-srcs",
        "//pkg/api/testapi:all-srcs",
        "//pkg/api/testing:all-srcs",
        "//pkg/api/unversioned:all-srcs",
        "//pkg/api/v1:all-srcs",
        "//pkg/api/validation:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
