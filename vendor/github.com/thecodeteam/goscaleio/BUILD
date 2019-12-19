load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "api.go",
        "certs.go",
        "device.go",
        "instance.go",
        "protectiondomain.go",
        "scsiinitiator.go",
        "sdc.go",
        "sds.go",
        "storagepool.go",
        "system.go",
        "user.go",
        "volume.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/thecodeteam/goscaleio",
    importpath = "github.com/thecodeteam/goscaleio",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/sirupsen/logrus:go_default_library",
        "//vendor/github.com/thecodeteam/goscaleio/types/v1:go_default_library",
    ],
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
        "//vendor/github.com/thecodeteam/goscaleio/types/v1:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
