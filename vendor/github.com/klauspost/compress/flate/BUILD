load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "deflate.go",
        "dict_decoder.go",
        "fast_encoder.go",
        "huffman_bit_writer.go",
        "huffman_code.go",
        "huffman_sortByFreq.go",
        "huffman_sortByLiteral.go",
        "inflate.go",
        "inflate_gen.go",
        "level1.go",
        "level2.go",
        "level3.go",
        "level4.go",
        "level5.go",
        "level6.go",
        "regmask_amd64.go",
        "regmask_other.go",
        "stateless.go",
        "token.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/klauspost/compress/flate",
    importpath = "github.com/klauspost/compress/flate",
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
