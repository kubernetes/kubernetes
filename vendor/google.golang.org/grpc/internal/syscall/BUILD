load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "syscall_linux.go",
        "syscall_nonlinux.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/grpc/internal/syscall",
    importpath = "google.golang.org/grpc/internal/syscall",
    visibility = ["//vendor/google.golang.org/grpc:__subpackages__"],
    deps = select({
        "@io_bazel_rules_go//go/platform:aix": [
            "//vendor/google.golang.org/grpc/grpclog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/google.golang.org/grpc/grpclog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/google.golang.org/grpc/grpclog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/google.golang.org/grpc/grpclog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/google.golang.org/grpc/grpclog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:illumos": [
            "//vendor/google.golang.org/grpc/grpclog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:ios": [
            "//vendor/google.golang.org/grpc/grpclog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:js": [
            "//vendor/google.golang.org/grpc/grpclog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
            "//vendor/google.golang.org/grpc/grpclog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//vendor/google.golang.org/grpc/grpclog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/google.golang.org/grpc/grpclog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/google.golang.org/grpc/grpclog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/google.golang.org/grpc/grpclog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/google.golang.org/grpc/grpclog:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/google.golang.org/grpc/grpclog:go_default_library",
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
