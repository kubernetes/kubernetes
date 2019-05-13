package(default_visibility = ["//visibility:public"])

load("@io_k8s_repo_infra//defs:go.bzl", "go_genrule")
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
        "int32.go",
        "int64.go",
        "string.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apimachinery/pkg/util/sets",
    importpath = "k8s.io/apimachinery/pkg/util/sets",
)

# This rule makes all sorts of terrible assumptions that it's running inside k8s.io/kubernetes, even though it's part of k8s.io/apimachinery. :\
go_genrule(
    name = "set-gen",
    srcs = [
        "//hack/boilerplate:boilerplate.generatego.txt",
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
$(location //vendor/k8s.io/code-generator/cmd/set-gen) \
    --input-dirs k8s.io/kubernetes/vendor/k8s.io/apimachinery/pkg/util/sets/types \
    --output-base $$(dirname $$(dirname $(location :byte.go))) \
    --go-header-file $(location //hack/boilerplate:boilerplate.generatego.txt) \
    --output-package sets
    """,
    go_deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/sets/types:go_default_library",
    ],
    tools = [
        "//vendor/k8s.io/code-generator/cmd/set-gen",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["set_test.go"],
    embed = [":go_default_library"],
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
        "//staging/src/k8s.io/apimachinery/pkg/util/sets/types:all-srcs",
    ],
    tags = ["automanaged"],
)
