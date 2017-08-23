load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "0doc.go",
        "binc.go",
        "cbor.go",
        "decode.go",
        "decode_go.go",
        "decode_go14.go",
        "encode.go",
        "fast-path.generated.go",
        "gen.generated.go",
        "gen.go",
        "gen-helper.generated.go",
        "gen_15.go",
        "gen_16.go",
        "gen_17.go",
        "helper.go",
        "helper_internal.go",
        "helper_not_unsafe.go",
        "json.go",
        "msgpack.go",
        "noop.go",
        "prebuild.go",
        "rpc.go",
        "simple.go",
        "time.go",
    ],
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/ugorji/go/codec/codecgen:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
