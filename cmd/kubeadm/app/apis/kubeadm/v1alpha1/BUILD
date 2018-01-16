load("@io_bazel_rules_go//go:def.bzl", "go_library")

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
        "//pkg/kubelet/apis/kubeletconfig/v1alpha1:go_default_library",
        "//pkg/proxy/apis/kubeproxyconfig/scheme:go_default_library",
        "//pkg/proxy/apis/kubeproxyconfig/v1alpha1:go_default_library",
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
    visibility = ["//visibility:public"],
)
