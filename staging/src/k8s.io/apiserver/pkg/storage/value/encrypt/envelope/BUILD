package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "envelope.go",
        "grpc_service.go",
        "metrics.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope",
    importpath = "k8s.io/apiserver/pkg/storage/value/encrypt/envelope",
    deps = [
        "//staging/src/k8s.io/apiserver/pkg/storage/value:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        "//staging/src/k8s.io/component-base/metrics:go_default_library",
        "//staging/src/k8s.io/component-base/metrics/legacyregistry:go_default_library",
        "//vendor/github.com/hashicorp/golang-lru:go_default_library",
        "//vendor/golang.org/x/crypto/cryptobyte:go_default_library",
        "//vendor/google.golang.org/grpc:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "envelope_test.go",
        "grpc_service_unix_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apiserver/pkg/storage/value:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/aes:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//staging/src/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/testing:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//staging/src/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/testing:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//staging/src/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/testing:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//staging/src/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/testing:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:ios": [
            "//staging/src/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/testing:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//staging/src/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/testing:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//staging/src/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/testing:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//staging/src/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/testing:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//staging/src/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/testing:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//staging/src/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/testing:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//staging/src/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
            "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/testing:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/testing:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:all-srcs",
    ],
    tags = ["automanaged"],
)
