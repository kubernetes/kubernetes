package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load("@io_kubernetes_build//defs:go.bzl", "go_genrule")
load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "byte.go",
        "doc.go",
        "empty.go",
        "int.go",
        "int64.go",
        "string.go",
    ],
    tags = ["automanaged"],
)

go_genrule(
    name = "set-gen",
    srcs = [
        "//hack/boilerplate:boilerplate.go.txt",
    ],
    outs = [
        "byte.go",
        "doc.go",
        "empty.go",
        "int.go",
        "int64.go",
        "string.go",
    ],
    cmd = """
$(location //cmd/libs/go2idl/set-gen) \
    --input-dirs ./vendor/k8s.io/apimachinery/pkg/util/sets/types \
    --output-base $(GENDIR)/vendor/k8s.io/apimachinery/pkg/util \
    --go-header-file $(location //hack/boilerplate:boilerplate.go.txt) \
    --output-package sets
    """,
    go_deps = [
        "//vendor/k8s.io/apimachinery/pkg/util/sets/types:go_default_library",
    ],
    tools = [
        "//cmd/libs/go2idl/set-gen",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["set_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
)
