load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "ascii.go",
        "proxy.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "tc.go",
            "term.go",
            "winsize.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "tc.go",
            "term.go",
            "termios_bsd.go",
            "winsize.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "tc.go",
            "term.go",
            "winsize.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "tc.go",
            "term.go",
            "termios_bsd.go",
            "winsize.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "tc.go",
            "term.go",
            "termios_linux.go",
            "winsize.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "tc.go",
            "term.go",
            "winsize.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "tc.go",
            "term.go",
            "winsize.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "tc.go",
            "term.go",
            "termios_bsd.go",
            "winsize.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "tc.go",
            "term.go",
            "winsize.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "tc.go",
            "tc_solaris_cgo.go",
            "term.go",
            "winsize_solaris_cgo.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "term_windows.go",
        ],
        "//conditions:default": [],
    }),
    cgo = True,
    importpath = "github.com/docker/docker/pkg/term",
    visibility = ["//visibility:public"],
    deps = select({
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
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/github.com/Azure/go-ansiterm/winterm:go_default_library",
            "//vendor/github.com/docker/docker/pkg/term/windows:go_default_library",
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
        "//vendor/github.com/docker/docker/pkg/term/windows:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
