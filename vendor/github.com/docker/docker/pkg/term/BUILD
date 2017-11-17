load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "ascii.go",
        "proxy.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "tc.go",
            "term.go",
            "termios_bsd.go",
            "winsize.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "tc.go",
            "term.go",
            "termios_linux.go",
            "winsize.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "term_windows.go",
        ],
        "//conditions:default": [],
    }),
    cgo = True,
    importpath = "github.com/docker/docker/pkg/term",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
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
