load("//build:deepcopy.bzl", "k8s_deepcopy", "k8s_defaulter", "k8s_conversion")

package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "defaults.go",
        "doc.go",
        "register.go",
        "types.go",
        "zz_generated.conversion.go",
        "zz_generated.deepcopy.go",
        "zz_generated.defaults.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/proxy/apis/kubeproxyconfig/v1alpha1",
    deps = [
        "//pkg/kubelet/qos:go_default_library",
        "//pkg/master/ports:go_default_library",
        "//pkg/proxy/apis/kubeproxyconfig:go_default_library",
        "//pkg/util/pointer:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
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

k8s_deepcopy(outs = ["zz_generated.deepcopy.go"])

k8s_defaulter(outs = ["zz_generated.defaults.go"])

k8s_conversion(outs = ["zz_generated.conversion.go"])
