load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "blockopt.go",
        "builder.go",
        "const.go",
        "create.go",
        "doc.go",
        "dom.go",
        "emit.go",
        "exits.go",
        "func.go",
        "html.go",
        "identical.go",
        "identical_17.go",
        "lift.go",
        "lvalue.go",
        "methods.go",
        "mode.go",
        "print.go",
        "sanity.go",
        "source.go",
        "ssa.go",
        "util.go",
        "wrappers.go",
        "write.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/honnef.co/go/tools/ir",
    importpath = "honnef.co/go/tools/ir",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/tools/go/ast/astutil:go_default_library",
        "//vendor/golang.org/x/tools/go/types/typeutil:go_default_library",
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
        "//vendor/honnef.co/go/tools/ir/irutil:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
