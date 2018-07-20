load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "graphbench_unix.go",
        "graphtest_unix.go",
        "graphtest_windows.go",
        "testutil.go",
        "testutil_unix.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/daemon/graphdriver/graphtest",
    importpath = "github.com/docker/docker/daemon/graphdriver/graphtest",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/containerd/continuity/driver:go_default_library",
        "//vendor/github.com/docker/docker/daemon/graphdriver:go_default_library",
        "//vendor/github.com/docker/docker/pkg/archive:go_default_library",
        "//vendor/github.com/docker/docker/pkg/stringid:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/github.com/docker/docker/daemon/graphdriver/quota:go_default_library",
            "//vendor/github.com/docker/go-units:go_default_library",
            "//vendor/github.com/stretchr/testify/assert:go_default_library",
            "//vendor/github.com/stretchr/testify/require:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/docker/docker/daemon/graphdriver/quota:go_default_library",
            "//vendor/github.com/docker/go-units:go_default_library",
            "//vendor/github.com/stretchr/testify/assert:go_default_library",
            "//vendor/github.com/stretchr/testify/require:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
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
