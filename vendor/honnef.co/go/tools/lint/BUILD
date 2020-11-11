load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "lint.go",
        "runner.go",
        "stats.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/honnef.co/go/tools/lint",
    importpath = "honnef.co/go/tools/lint",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/tools/go/analysis:go_default_library",
        "//vendor/golang.org/x/tools/go/packages:go_default_library",
        "//vendor/golang.org/x/tools/go/types/objectpath:go_default_library",
        "//vendor/honnef.co/go/tools/config:go_default_library",
        "//vendor/honnef.co/go/tools/facts:go_default_library",
        "//vendor/honnef.co/go/tools/internal/cache:go_default_library",
        "//vendor/honnef.co/go/tools/loader:go_default_library",
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
        "//vendor/honnef.co/go/tools/lint/lintdsl:all-srcs",
        "//vendor/honnef.co/go/tools/lint/lintutil:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
