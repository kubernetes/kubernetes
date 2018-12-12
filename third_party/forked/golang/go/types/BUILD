licenses(["notice"])

load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "api.go",
        "assignments.go",
        "builtins.go",
        "call.go",
        "check.go",
        "conversions.go",
        "decl.go",
        "errors.go",
        "eval.go",
        "expr.go",
        "exprstring.go",
        "initorder.go",
        "labels.go",
        "lookup.go",
        "methodset.go",
        "object.go",
        "objset.go",
        "operand.go",
        "ordering.go",
        "package.go",
        "predicates.go",
        "resolver.go",
        "return.go",
        "scope.go",
        "selection.go",
        "sizes.go",
        "stmt.go",
        "type.go",
        "typestring.go",
        "typexpr.go",
        "universe.go",
    ],
    importpath = "k8s.io/kubernetes/third_party/forked/golang/go/types",
    visibility = ["//visibility:public"],
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
