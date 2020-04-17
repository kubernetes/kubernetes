load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "callback.go",
        "cgo.go",
        "errors.go",
        "hcs.go",
        "log.go",
        "process.go",
        "system.go",
        "utils.go",
        "waithelper.go",
        "watcher.go",
        "zsyscall_windows.go",
    ],
    cgo = True,
    importmap = "k8s.io/kubernetes/vendor/github.com/Microsoft/hcsshim/internal/hcs",
    importpath = "github.com/Microsoft/hcsshim/internal/hcs",
    visibility = ["//vendor/github.com/Microsoft/hcsshim:__subpackages__"],
    deps = [
        "//vendor/github.com/Microsoft/go-winio:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/interop:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/logfields:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/schema1:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/timeout:go_default_library",
        "//vendor/github.com/sirupsen/logrus:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:windows": [
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
