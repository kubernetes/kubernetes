load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "device_setup.go",
        "deviceset.go",
        "devmapper_doc.go",
        "driver.go",
        "mount.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/daemon/graphdriver/devmapper",
    importpath = "github.com/docker/docker/daemon/graphdriver/devmapper",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/github.com/sirupsen/logrus:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/docker/docker/daemon/graphdriver:go_default_library",
            "//vendor/github.com/docker/docker/dockerversion:go_default_library",
            "//vendor/github.com/docker/docker/pkg/containerfs:go_default_library",
            "//vendor/github.com/docker/docker/pkg/devicemapper:go_default_library",
            "//vendor/github.com/docker/docker/pkg/dmesg:go_default_library",
            "//vendor/github.com/docker/docker/pkg/idtools:go_default_library",
            "//vendor/github.com/docker/docker/pkg/locker:go_default_library",
            "//vendor/github.com/docker/docker/pkg/loopback:go_default_library",
            "//vendor/github.com/docker/docker/pkg/mount:go_default_library",
            "//vendor/github.com/docker/docker/pkg/parsers:go_default_library",
            "//vendor/github.com/docker/docker/pkg/parsers/kernel:go_default_library",
            "//vendor/github.com/docker/docker/pkg/system:go_default_library",
            "//vendor/github.com/docker/go-units:go_default_library",
            "//vendor/github.com/opencontainers/selinux/go-selinux/label:go_default_library",
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
