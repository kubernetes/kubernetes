load("//build:deepcopy.bzl", "k8s_deepcopy", "k8s_conversion")

package(default_visibility = ["//visibility:public"])

load(
    "//build:go.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "register_test.go",
        "roundtrip_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/api/testing/roundtrip:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/fuzzer:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/diff:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "conversion.go",
        "doc.go",
        "register.go",
        "types.go",
        "zz_generated.conversion.go",
        "zz_generated.deepcopy.go",
    ],
    importpath = "k8s.io/apimachinery/pkg/apis/meta/internalversion",
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1beta1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/conversion:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
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

k8s_deepcopy(outs = ["zz_generated.deepcopy.go"])

k8s_conversion(outs = ["zz_generated.conversion.go"])
