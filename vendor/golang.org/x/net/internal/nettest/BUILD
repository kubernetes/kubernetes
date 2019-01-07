load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "helper_bsd.go",
        "helper_nobsd.go",
        "helper_posix.go",
        "helper_stub.go",
        "helper_unix.go",
        "helper_windows.go",
        "interface.go",
        "rlimit.go",
        "stack.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/net/internal/nettest",
    importpath = "golang.org/x/net/internal/nettest",
    visibility = ["//vendor/golang.org/x/net:__subpackages__"],
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
