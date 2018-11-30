load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "dot.go",
        "sem.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gonum.org/v1/gonum/graph/formats/dot",
    importpath = "gonum.org/v1/gonum/graph/formats/dot",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/gonum.org/v1/gonum/graph/formats/dot/ast:go_default_library",
        "//vendor/gonum.org/v1/gonum/graph/formats/dot/internal/lexer:go_default_library",
        "//vendor/gonum.org/v1/gonum/graph/formats/dot/internal/parser:go_default_library",
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
        "//vendor/gonum.org/v1/gonum/graph/formats/dot/ast:all-srcs",
        "//vendor/gonum.org/v1/gonum/graph/formats/dot/internal/astx:all-srcs",
        "//vendor/gonum.org/v1/gonum/graph/formats/dot/internal/errors:all-srcs",
        "//vendor/gonum.org/v1/gonum/graph/formats/dot/internal/lexer:all-srcs",
        "//vendor/gonum.org/v1/gonum/graph/formats/dot/internal/parser:all-srcs",
        "//vendor/gonum.org/v1/gonum/graph/formats/dot/internal/token:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
