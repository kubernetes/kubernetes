load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "buddyinfo.go",
        "doc.go",
        "fs.go",
        "ipvs.go",
        "mdstat.go",
        "mountstats.go",
        "net_dev.go",
        "net_unix.go",
        "proc.go",
        "proc_io.go",
        "proc_limits.go",
        "proc_ns.go",
        "proc_psi.go",
        "proc_stat.go",
        "proc_status.go",
        "stat.go",
        "xfrm.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/prometheus/procfs",
    importpath = "github.com/prometheus/procfs",
    visibility = ["//staging/src/k8s.io/component-base/metrics:prometheus_import_allow_list"],
    deps = ["//vendor/github.com/prometheus/procfs/internal/fs:go_default_library"],
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
        "//vendor/github.com/prometheus/procfs/internal/fs:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
