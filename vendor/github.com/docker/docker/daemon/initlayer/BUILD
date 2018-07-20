load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "setup_unix.go",
        "setup_windows.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/daemon/initlayer",
    importpath = "github.com/docker/docker/daemon/initlayer",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/github.com/docker/docker/pkg/containerfs:go_default_library",
            "//vendor/github.com/docker/docker/pkg/idtools:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/docker/docker/pkg/containerfs:go_default_library",
            "//vendor/github.com/docker/docker/pkg/idtools:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/docker/docker/pkg/containerfs:go_default_library",
            "//vendor/github.com/docker/docker/pkg/idtools:go_default_library",
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
