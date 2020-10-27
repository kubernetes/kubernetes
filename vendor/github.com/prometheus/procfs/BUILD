load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "arp.go",
        "buddyinfo.go",
        "cpuinfo.go",
        "cpuinfo_arm.go",
        "cpuinfo_arm64.go",
        "cpuinfo_default.go",
        "cpuinfo_mips.go",
        "cpuinfo_mips64.go",
        "cpuinfo_mips64le.go",
        "cpuinfo_mipsle.go",
        "cpuinfo_ppc64.go",
        "cpuinfo_ppc64le.go",
        "cpuinfo_s390x.go",
        "crypto.go",
        "doc.go",
        "fs.go",
        "fscache.go",
        "ipvs.go",
        "kernel_random.go",
        "loadavg.go",
        "mdstat.go",
        "meminfo.go",
        "mountinfo.go",
        "mountstats.go",
        "net_conntrackstat.go",
        "net_dev.go",
        "net_sockstat.go",
        "net_softnet.go",
        "net_udp.go",
        "net_unix.go",
        "proc.go",
        "proc_cgroup.go",
        "proc_environ.go",
        "proc_fdinfo.go",
        "proc_io.go",
        "proc_limits.go",
        "proc_maps.go",
        "proc_ns.go",
        "proc_psi.go",
        "proc_smaps.go",
        "proc_stat.go",
        "proc_status.go",
        "schedstat.go",
        "stat.go",
        "swaps.go",
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
    ] + select({
        "@io_bazel_rules_go//go/platform:aix": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:ios": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/prometheus/procfs/internal/fs:all-srcs",
        "//vendor/github.com/prometheus/procfs/internal/util:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
