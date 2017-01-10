package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        ":set-gen",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["set_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
)

genrule(
    name = "set-gen",
    srcs = [
        "//pkg/util/sets/types:types.go",
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
    --input-dirs ./pkg/util/sets/types \
    --output-base $(GENDIR)/pkg/util \
    --go-header-file $(location //hack/boilerplate:boilerplate.go.txt) \
    --output-package sets
    """,
    tools = [
        "//cmd/libs/go2idl/set-gen",
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
        "//pkg/util/sets/types:all-srcs",
    ],
    tags = ["automanaged"],
)
