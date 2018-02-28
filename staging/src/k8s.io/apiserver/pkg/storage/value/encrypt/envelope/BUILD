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
    ],
    importpath = "k8s.io/apiserver/pkg/storage/value/encrypt/envelope",
    deps = [
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/hashicorp/golang-lru:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/google.golang.org/grpc:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/value:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "envelope_test.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "grpc_service_unix_test.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "grpc_service_unix_test.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "grpc_service_unix_test.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "grpc_service_unix_test.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "grpc_service_unix_test.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "grpc_service_unix_test.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "grpc_service_unix_test.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "grpc_service_unix_test.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "grpc_service_unix_test.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "grpc_service_unix_test.go",
        ],
        "//conditions:default": [],
    }),
    embed = [":go_default_library"],
    deps = [
        "//vendor/k8s.io/apiserver/pkg/storage/value:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/aes:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/google.golang.org/grpc:go_default_library",
            "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:go_default_library",
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
        "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope/v1beta1:all-srcs",
    ],
    tags = ["automanaged"],
)
