load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "client.go",
        "clientconfig.go",
        "defaults.go",
        "dns.go",
        "dnssec.go",
        "dnssec_keygen.go",
        "dnssec_keyscan.go",
        "dnssec_privkey.go",
        "doc.go",
        "edns.go",
        "format.go",
        "generate.go",
        "labels.go",
        "msg.go",
        "msg_helpers.go",
        "nsecx.go",
        "privaterr.go",
        "rawmsg.go",
        "reverse.go",
        "sanitize.go",
        "scan.go",
        "scan_rr.go",
        "scanner.go",
        "server.go",
        "sig0.go",
        "singleinflight.go",
        "tlsa.go",
        "tsig.go",
        "types.go",
        "update.go",
        "xfr.go",
        "zmsg.go",
        "ztypes.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "udp.go",
            "udp_other.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "udp.go",
            "udp_other.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "udp.go",
            "udp_other.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "udp.go",
            "udp_other.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "udp.go",
            "udp_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "udp.go",
            "udp_other.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "udp.go",
            "udp_other.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "udp.go",
            "udp_other.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "udp_plan9.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "udp.go",
            "udp_other.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "udp_other.go",
            "udp_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/miekg/dns",
    visibility = ["//visibility:public"],
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
