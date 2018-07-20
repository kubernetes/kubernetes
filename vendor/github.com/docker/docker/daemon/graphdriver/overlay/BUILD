load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "overlay.go",
        "overlay_unsupported.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/daemon/graphdriver/overlay",
    importpath = "github.com/docker/docker/daemon/graphdriver/overlay",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/docker/docker/daemon/graphdriver:go_default_library",
            "//vendor/github.com/docker/docker/daemon/graphdriver/copy:go_default_library",
            "//vendor/github.com/docker/docker/daemon/graphdriver/overlayutils:go_default_library",
            "//vendor/github.com/docker/docker/pkg/archive:go_default_library",
            "//vendor/github.com/docker/docker/pkg/containerfs:go_default_library",
            "//vendor/github.com/docker/docker/pkg/fsutils:go_default_library",
            "//vendor/github.com/docker/docker/pkg/idtools:go_default_library",
            "//vendor/github.com/docker/docker/pkg/locker:go_default_library",
            "//vendor/github.com/docker/docker/pkg/mount:go_default_library",
            "//vendor/github.com/docker/docker/pkg/system:go_default_library",
            "//vendor/github.com/opencontainers/selinux/go-selinux/label:go_default_library",
            "//vendor/github.com/sirupsen/logrus:go_default_library",
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
