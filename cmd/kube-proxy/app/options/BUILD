load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "options.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "init_others.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "init_others.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "init_others.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "init_others.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "init_others.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "init_others.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "init_others.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "init_others.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "init_others.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "init_others.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "init_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/cmd/kube-proxy/app/options",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/apis/componentconfig:go_default_library",
        "//pkg/master/ports:go_default_library",
        "//pkg/proxy/apis/kubeproxyconfig:go_default_library",
        "//pkg/proxy/apis/kubeproxyconfig/scheme:go_default_library",
        "//pkg/proxy/apis/kubeproxyconfig/v1alpha1:go_default_library",
        "//pkg/util/pointer:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer/json:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/flag:go_default_library",
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
