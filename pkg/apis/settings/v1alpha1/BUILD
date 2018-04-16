load("//build:deepcopy.bzl", "k8s_defaulter", "k8s_conversion")

package(default_visibility = ["//visibility:public"])

load(
    "//build:go.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "register.go",
        "zz_generated.conversion.go",
        "zz_generated.defaults.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/apis/settings/v1alpha1",
    deps = [
        "//pkg/apis/core:go_default_library",
        "//pkg/apis/core/v1:go_default_library",
        "//pkg/apis/settings:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/api/settings/v1alpha1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/conversion:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
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

k8s_conversion(outs = ["zz_generated.conversion.go"])

k8s_defaulter(outs = ["zz_generated.defaults.go"])
