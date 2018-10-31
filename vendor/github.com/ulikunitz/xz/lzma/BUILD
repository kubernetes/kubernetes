load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bintree.go",
        "bitops.go",
        "breader.go",
        "buffer.go",
        "bytewriter.go",
        "decoder.go",
        "decoderdict.go",
        "directcodec.go",
        "distcodec.go",
        "encoder.go",
        "encoderdict.go",
        "hashtable.go",
        "header.go",
        "header2.go",
        "lengthcodec.go",
        "literalcodec.go",
        "matchalgorithm.go",
        "operation.go",
        "prob.go",
        "properties.go",
        "rangecodec.go",
        "reader.go",
        "reader2.go",
        "state.go",
        "treecodecs.go",
        "writer.go",
        "writer2.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/ulikunitz/xz/lzma",
    importpath = "github.com/ulikunitz/xz/lzma",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/ulikunitz/xz/internal/hash:go_default_library",
        "//vendor/github.com/ulikunitz/xz/internal/xlog:go_default_library",
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
