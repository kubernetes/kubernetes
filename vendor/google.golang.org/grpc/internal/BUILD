load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["internal.go"],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/grpc/internal",
    importpath = "google.golang.org/grpc/internal",
    visibility = ["//vendor/google.golang.org/grpc:__subpackages__"],
    deps = ["//vendor/google.golang.org/grpc/connectivity:go_default_library"],
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
        "//vendor/google.golang.org/grpc/internal/backoff:all-srcs",
        "//vendor/google.golang.org/grpc/internal/balancerload:all-srcs",
        "//vendor/google.golang.org/grpc/internal/binarylog:all-srcs",
        "//vendor/google.golang.org/grpc/internal/buffer:all-srcs",
        "//vendor/google.golang.org/grpc/internal/channelz:all-srcs",
        "//vendor/google.golang.org/grpc/internal/envconfig:all-srcs",
        "//vendor/google.golang.org/grpc/internal/grpcrand:all-srcs",
        "//vendor/google.golang.org/grpc/internal/grpcsync:all-srcs",
        "//vendor/google.golang.org/grpc/internal/resolver/dns:all-srcs",
        "//vendor/google.golang.org/grpc/internal/resolver/passthrough:all-srcs",
        "//vendor/google.golang.org/grpc/internal/syscall:all-srcs",
        "//vendor/google.golang.org/grpc/internal/transport:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
