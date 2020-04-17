package(default_visibility = ["//visibility:public"])

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
        "//pkg/util/async:all-srcs",
        "//pkg/util/bandwidth:all-srcs",
        "//pkg/util/config:all-srcs",
        "//pkg/util/configz:all-srcs",
        "//pkg/util/conntrack:all-srcs",
        "//pkg/util/coverage:all-srcs",
        "//pkg/util/ebtables:all-srcs",
        "//pkg/util/env:all-srcs",
        "//pkg/util/filesystem:all-srcs",
        "//pkg/util/flag:all-srcs",
        "//pkg/util/flock:all-srcs",
        "//pkg/util/goroutinemap:all-srcs",
        "//pkg/util/hash:all-srcs",
        "//pkg/util/interrupt:all-srcs",
        "//pkg/util/ipconfig:all-srcs",
        "//pkg/util/ipset:all-srcs",
        "//pkg/util/iptables:all-srcs",
        "//pkg/util/ipvs:all-srcs",
        "//pkg/util/labels:all-srcs",
        "//pkg/util/maps:all-srcs",
        "//pkg/util/netsh:all-srcs",
        "//pkg/util/node:all-srcs",
        "//pkg/util/oom:all-srcs",
        "//pkg/util/parsers:all-srcs",
        "//pkg/util/pod:all-srcs",
        "//pkg/util/procfs:all-srcs",
        "//pkg/util/removeall:all-srcs",
        "//pkg/util/resizefs:all-srcs",
        "//pkg/util/rlimit:all-srcs",
        "//pkg/util/selinux:all-srcs",
        "//pkg/util/slice:all-srcs",
        "//pkg/util/sysctl:all-srcs",
        "//pkg/util/tail:all-srcs",
        "//pkg/util/taints:all-srcs",
        "//pkg/util/tolerations:all-srcs",
    ],
    tags = ["automanaged"],
)

sh_test(
    name = "verify-util-pkg",
    size = "small",
    srcs = ["verify-util-pkg.sh"],
    data = glob(["*.go"]),
)
