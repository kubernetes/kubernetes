load("//build:deepcopy.bzl", "k8s_deepcopy", "k8s_defaulter", "k8s_conversion")
load("//build:go.bzl", "go_library")

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
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "defaults_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "defaults_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "defaults_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "defaults_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "defaults_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "defaults_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "defaults_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "defaults_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "defaults_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "defaults_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "defaults_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/constants:go_default_library",
        "//cmd/kubeadm/app/features:go_default_library",
        "//pkg/kubelet/apis/kubeletconfig/scheme:go_default_library",
        "//pkg/kubelet/apis/kubeletconfig/v1beta1:go_default_library",
        "//pkg/proxy/apis/kubeproxyconfig/scheme:go_default_library",
        "//pkg/proxy/apis/kubeproxyconfig/v1alpha1:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
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
    visibility = ["//visibility:public"],
)

k8s_deepcopy(outs = ["zz_generated.deepcopy.go"])

k8s_conversion(outs = ["zz_generated.conversion.go"])

k8s_defaulter(outs = ["zz_generated.defaults.go"])
