load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "callback.go",
        "errors.go",
        "process.go",
        "service.go",
        "syscall.go",
        "system.go",
        "utils.go",
        "waithelper.go",
        "zsyscall_windows.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/Microsoft/hcsshim/internal/hcs",
    importpath = "github.com/Microsoft/hcsshim/internal/hcs",
    visibility = ["//vendor/github.com/Microsoft/hcsshim:__subpackages__"],
    deps = [
        "//vendor/github.com/Microsoft/go-winio:go_default_library",
        "//vendor/github.com/Microsoft/go-winio/vhd:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/cow:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/interop:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/log:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/logfields:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/oc:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/schema1:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/schema2:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/timeout:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/vmcompute:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/github.com/sirupsen/logrus:go_default_library",
        "//vendor/go.opencensus.io/trace:go_default_library",
        "//vendor/golang.org/x/sys/windows:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
