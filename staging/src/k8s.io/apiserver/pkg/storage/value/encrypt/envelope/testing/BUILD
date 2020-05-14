load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["kms_plugin_mock.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/testing",
    importpath = "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/testing",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/google.golang.org/grpc/codes:go_default_library",
            "//vendor/google.golang.org/grpc/status:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/google.golang.org/grpc/codes:go_default_library",
            "//vendor/google.golang.org/grpc/status:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/google.golang.org/grpc/codes:go_default_library",
            "//vendor/google.golang.org/grpc/status:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/google.golang.org/grpc/codes:go_default_library",
            "//vendor/google.golang.org/grpc/status:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:ios": [
            "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/google.golang.org/grpc/codes:go_default_library",
            "//vendor/google.golang.org/grpc/status:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/google.golang.org/grpc/codes:go_default_library",
            "//vendor/google.golang.org/grpc/status:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/google.golang.org/grpc/codes:go_default_library",
            "//vendor/google.golang.org/grpc/status:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/google.golang.org/grpc/codes:go_default_library",
            "//vendor/google.golang.org/grpc/status:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/google.golang.org/grpc/codes:go_default_library",
            "//vendor/google.golang.org/grpc/status:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/google.golang.org/grpc/codes:go_default_library",
            "//vendor/google.golang.org/grpc/status:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/google.golang.org/grpc/codes:go_default_library",
            "//vendor/google.golang.org/grpc/status:go_default_library",
            "//vendor/k8s.io/klog/v2:go_default_library",
        ],
        "//conditions:default": [],
    }),
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
