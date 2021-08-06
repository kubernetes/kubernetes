load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "term.go",
        "term_plan9.go",
        "term_solaris.go",
        "term_unix.go",
        "term_unix_aix.go",
        "term_unix_bsd.go",
        "term_unix_linux.go",
        "term_unix_zos.go",
        "term_unsupported.go",
        "term_windows.go",
        "terminal.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/term",
    importpath = "golang.org/x/term",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/sys/unix:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:plan9": [
            "//vendor/golang.org/x/sys/plan9:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//vendor/golang.org/x/sys/windows:go_default_library",
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
