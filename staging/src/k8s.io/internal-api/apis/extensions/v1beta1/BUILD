package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

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
    importmap = "k8s.io/kubernetes/vendor/k8s.io/internal-api/apis/extensions/v1beta1",
    importpath = "k8s.io/internal-api/apis/extensions/v1beta1",
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/extensions/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/conversion:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/apps:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/autoscaling:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/core:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/core/v1:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/extensions:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/networking:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/policy:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["defaults_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/extensions/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
        "//staging/src/k8s.io/internal-api/api/legacyscheme:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/core:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/core/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/extensions/install:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
