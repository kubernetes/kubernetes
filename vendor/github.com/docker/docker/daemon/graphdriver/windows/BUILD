load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["windows.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/daemon/graphdriver/windows",
    importpath = "github.com/docker/docker/daemon/graphdriver/windows",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/Microsoft/go-winio:go_default_library",
            "//vendor/github.com/Microsoft/go-winio/archive/tar:go_default_library",
            "//vendor/github.com/Microsoft/go-winio/backuptar:go_default_library",
            "//vendor/github.com/Microsoft/hcsshim:go_default_library",
            "//vendor/github.com/docker/docker/daemon/graphdriver:go_default_library",
            "//vendor/github.com/docker/docker/pkg/archive:go_default_library",
            "//vendor/github.com/docker/docker/pkg/containerfs:go_default_library",
            "//vendor/github.com/docker/docker/pkg/idtools:go_default_library",
            "//vendor/github.com/docker/docker/pkg/ioutils:go_default_library",
            "//vendor/github.com/docker/docker/pkg/longpath:go_default_library",
            "//vendor/github.com/docker/docker/pkg/reexec:go_default_library",
            "//vendor/github.com/docker/docker/pkg/system:go_default_library",
            "//vendor/github.com/docker/go-units:go_default_library",
            "//vendor/github.com/sirupsen/logrus:go_default_library",
            "//vendor/golang.org/x/sys/windows:go_default_library",
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
