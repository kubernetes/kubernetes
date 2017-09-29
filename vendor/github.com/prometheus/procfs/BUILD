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
        "proc.go",
        "proc_io.go",
        "proc_limits.go",
        "proc_stat.go",
        "stat.go",
        "xfrm.go",
    ],
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/prometheus/procfs/xfs:go_default_library"],
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
        "//vendor/github.com/prometheus/procfs/xfs:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
