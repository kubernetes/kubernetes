package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

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
        "udp.go",
        "udp_linux.go",
        "update.go",
        "xfr.go",
        "zmsg.go",
        "ztypes.go",
    ],
    tags = ["automanaged"],
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
)
