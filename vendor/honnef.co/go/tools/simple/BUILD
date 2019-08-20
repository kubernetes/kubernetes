load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "analysis.go",
        "doc.go",
        "lint.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/honnef.co/go/tools/simple",
    importpath = "honnef.co/go/tools/simple",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/tools/go/analysis:go_default_library",
        "//vendor/golang.org/x/tools/go/analysis/passes/inspect:go_default_library",
        "//vendor/golang.org/x/tools/go/ast/inspector:go_default_library",
        "//vendor/golang.org/x/tools/go/types/typeutil:go_default_library",
        "//vendor/honnef.co/go/tools/arg:go_default_library",
        "//vendor/honnef.co/go/tools/facts:go_default_library",
        "//vendor/honnef.co/go/tools/internal/passes/buildssa:go_default_library",
        "//vendor/honnef.co/go/tools/internal/sharedcheck:go_default_library",
        "//vendor/honnef.co/go/tools/lint:go_default_library",
        "//vendor/honnef.co/go/tools/lint/lintdsl:go_default_library",
        "//vendor/honnef.co/go/tools/lint/lintutil:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
