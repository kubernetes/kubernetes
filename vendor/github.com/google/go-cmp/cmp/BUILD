load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "compare.go",
        "export_unsafe.go",
        "options.go",
        "path.go",
        "report.go",
        "report_compare.go",
        "report_reflect.go",
        "report_slices.go",
        "report_text.go",
        "report_value.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/google/go-cmp/cmp",
    importpath = "github.com/google/go-cmp/cmp",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/google/go-cmp/cmp/internal/diff:go_default_library",
        "//vendor/github.com/google/go-cmp/cmp/internal/flags:go_default_library",
        "//vendor/github.com/google/go-cmp/cmp/internal/function:go_default_library",
        "//vendor/github.com/google/go-cmp/cmp/internal/value:go_default_library",
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
        "//vendor/github.com/google/go-cmp/cmp/cmpopts:all-srcs",
        "//vendor/github.com/google/go-cmp/cmp/internal/diff:all-srcs",
        "//vendor/github.com/google/go-cmp/cmp/internal/flags:all-srcs",
        "//vendor/github.com/google/go-cmp/cmp/internal/function:all-srcs",
        "//vendor/github.com/google/go-cmp/cmp/internal/value:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
