package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "fake_image_service.go",
        "fake_runtime.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "endpoint.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "endpoint.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "endpoint.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "endpoint.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "endpoint.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "endpoint.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "endpoint.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "endpoint.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "endpoint.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "endpoint.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "endpoint_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/kubelet/remote/fake",
    tags = ["automanaged"],
    deps = [
        "//pkg/kubelet/apis/cri/runtime/v1alpha3:go_default_library",
        "//pkg/kubelet/apis/cri/testing:go_default_library",
        "//pkg/kubelet/util:go_default_library",
        "//vendor/google.golang.org/grpc:go_default_library",
        "//vendor/k8s.io/utils/exec:go_default_library",
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
