load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "acceptfunc.go",
        "client.go",
        "clientconfig.go",
        "dane.go",
        "defaults.go",
        "dns.go",
        "dnssec.go",
        "dnssec_keygen.go",
        "dnssec_keyscan.go",
        "dnssec_privkey.go",
        "doc.go",
        "duplicate.go",
        "edns.go",
        "format.go",
        "generate.go",
        "labels.go",
        "listen_go111.go",
        "listen_go_not111.go",
        "msg.go",
        "msg_helpers.go",
        "nsecx.go",
        "privaterr.go",
        "reverse.go",
        "sanitize.go",
        "scan.go",
        "scan_rr.go",
        "serve_mux.go",
        "server.go",
        "sig0.go",
        "singleinflight.go",
        "smimea.go",
        "tlsa.go",
        "tsig.go",
        "types.go",
        "udp.go",
        "udp_windows.go",
        "update.go",
        "version.go",
        "xfr.go",
        "zduplicate.go",
        "zmsg.go",
        "ztypes.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/miekg/dns",
    importpath = "github.com/miekg/dns",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/crypto/ed25519:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/golang.org/x/net/ipv4:go_default_library",
            "//vendor/golang.org/x/net/ipv6:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "//vendor/golang.org/x/net/ipv4:go_default_library",
            "//vendor/golang.org/x/net/ipv6:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "//vendor/golang.org/x/net/ipv4:go_default_library",
            "//vendor/golang.org/x/net/ipv6:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "//vendor/golang.org/x/net/ipv4:go_default_library",
            "//vendor/golang.org/x/net/ipv6:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/golang.org/x/net/ipv4:go_default_library",
            "//vendor/golang.org/x/net/ipv6:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//vendor/golang.org/x/net/ipv4:go_default_library",
            "//vendor/golang.org/x/net/ipv6:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/golang.org/x/net/ipv4:go_default_library",
            "//vendor/golang.org/x/net/ipv6:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/golang.org/x/net/ipv4:go_default_library",
            "//vendor/golang.org/x/net/ipv6:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/golang.org/x/net/ipv4:go_default_library",
            "//vendor/golang.org/x/net/ipv6:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/golang.org/x/net/ipv4:go_default_library",
            "//vendor/golang.org/x/net/ipv6:go_default_library",
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
