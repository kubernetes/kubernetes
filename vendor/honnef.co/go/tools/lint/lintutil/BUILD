load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "stats.go",
        "stats_bsd.go",
        "stats_posix.go",
        "util.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/honnef.co/go/tools/lint/lintutil",
    importpath = "honnef.co/go/tools/lint/lintutil",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/tools/go/analysis:go_default_library",
        "//vendor/golang.org/x/tools/go/buildutil:go_default_library",
        "//vendor/golang.org/x/tools/go/packages:go_default_library",
        "//vendor/honnef.co/go/tools/config:go_default_library",
        "//vendor/honnef.co/go/tools/internal/cache:go_default_library",
        "//vendor/honnef.co/go/tools/lint:go_default_library",
        "//vendor/honnef.co/go/tools/lint/lintutil/format:go_default_library",
        "//vendor/honnef.co/go/tools/version:go_default_library",
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
        "//vendor/honnef.co/go/tools/lint/lintutil/format:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
