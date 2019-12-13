load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "conversion.go",
        "defaults.go",
        "doc.go",
        "register.go",
        "zz_generated.conversion.go",
        "zz_generated.defaults.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/apis/core/v1",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/apis/apps:go_default_library",
        "//pkg/apis/core:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/util/parsers:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/conversion:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//vendor/k8s.io/utils/net:go_default_library",
        "//vendor/k8s.io/utils/pointer:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "conversion_test.go",
        "defaults_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/apis/apps:go_default_library",
        "//pkg/apis/apps/install:go_default_library",
        "//pkg/apis/core:go_default_library",
        "//pkg/apis/core/fuzzer:go_default_library",
        "//pkg/apis/core/install:go_default_library",
        "//pkg/features:go_default_library",
        "//staging/src/k8s.io/api/apps/v1:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/apitesting/fuzzer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/fuzzer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/component-base/featuregate/testing:go_default_library",
        "//vendor/k8s.io/utils/pointer:go_default_library",
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
        "//pkg/apis/core/v1/helper:all-srcs",
        "//pkg/apis/core/v1/validation:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
