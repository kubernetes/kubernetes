load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "arp.go",
        "buddyinfo.go",
        "cpuinfo.go",
        "crypto.go",
        "doc.go",
        "fs.go",
        "ipvs.go",
        "mdstat.go",
        "mountinfo.go",
        "mountstats.go",
        "net_dev.go",
        "net_softnet.go",
        "net_unix.go",
        "proc.go",
        "proc_environ.go",
        "proc_fdinfo.go",
        "proc_io.go",
        "proc_limits.go",
        "proc_ns.go",
        "proc_psi.go",
        "proc_stat.go",
        "proc_status.go",
        "schedstat.go",
        "stat.go",
        "vm.go",
        "xfrm.go",
        "zoneinfo.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/prometheus/procfs",
    importpath = "github.com/prometheus/procfs",
    visibility = ["//build/visible_to:vendor_githubcom_prometheus_CONSUMERS"],
    deps = [
        "//vendor/github.com/prometheus/procfs/internal/fs:go_default_library",
        "//vendor/github.com/prometheus/procfs/internal/util:go_default_library",
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
        "//vendor/github.com/prometheus/procfs/internal/fs:all-srcs",
        "//vendor/github.com/prometheus/procfs/internal/util:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
