load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "fix.go",
        "imports.go",
        "mod.go",
        "sortimports.go",
        "zstdlib.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/tools/internal/imports",
    importpath = "golang.org/x/tools/internal/imports",
    visibility = ["//vendor/golang.org/x/tools:__subpackages__"],
    deps = [
        "//vendor/golang.org/x/tools/go/ast/astutil:go_default_library",
        "//vendor/golang.org/x/tools/go/packages:go_default_library",
        "//vendor/golang.org/x/tools/internal/gopathwalk:go_default_library",
        "//vendor/golang.org/x/tools/internal/module:go_default_library",
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
