load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "lcow.go",
        "lcow_svm.go",
        "remotefs.go",
        "remotefs_file.go",
        "remotefs_filedriver.go",
        "remotefs_pathdriver.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/daemon/graphdriver/lcow",
    importpath = "github.com/docker/docker/daemon/graphdriver/lcow",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/Microsoft/hcsshim:go_default_library",
            "//vendor/github.com/Microsoft/opengcs/client:go_default_library",
            "//vendor/github.com/Microsoft/opengcs/service/gcsutils/remotefs:go_default_library",
            "//vendor/github.com/containerd/continuity/driver:go_default_library",
            "//vendor/github.com/containerd/continuity/pathdriver:go_default_library",
            "//vendor/github.com/docker/docker/daemon/graphdriver:go_default_library",
            "//vendor/github.com/docker/docker/pkg/archive:go_default_library",
            "//vendor/github.com/docker/docker/pkg/containerfs:go_default_library",
            "//vendor/github.com/docker/docker/pkg/idtools:go_default_library",
            "//vendor/github.com/docker/docker/pkg/ioutils:go_default_library",
            "//vendor/github.com/docker/docker/pkg/system:go_default_library",
            "//vendor/github.com/sirupsen/logrus:go_default_library",
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
