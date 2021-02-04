load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "attach.go",
        "destroy.go",
        "detach.go",
        "export.go",
        "format.go",
        "helpers.go",
        "import.go",
        "initialize.go",
        "mount.go",
        "setup.go",
        "storage.go",
        "zsyscall_windows.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/Microsoft/hcsshim/computestorage",
    importpath = "github.com/Microsoft/hcsshim/computestorage",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/Microsoft/go-winio/pkg/security:go_default_library",
        "//vendor/github.com/Microsoft/go-winio/vhd:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/interop:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/oc:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/internal/schema2:go_default_library",
        "//vendor/github.com/Microsoft/hcsshim/osversion:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
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
