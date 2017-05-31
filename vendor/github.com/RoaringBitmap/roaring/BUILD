package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "arraycontainer.go",
        "arraycontainer_gen.go",
        "bitmapcontainer.go",
        "bitmapcontainer_gen.go",
        "ctz.go",
        "fastaggregation.go",
        "popcnt.go",
        "popcnt_amd64.s",
        "popcnt_asm.go",
        "priorityqueue.go",
        "rle.go",
        "rle16.go",
        "rle16_gen.go",
        "rle_gen.go",
        "rlecommon.go",
        "rlei.go",
        "roaring.go",
        "roaringarray.go",
        "roaringarray_gen.go",
        "serialization.go",
        "serialization_littleendian.go",
        "setutil.go",
        "shortiterator.go",
        "util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/github.com/glycerine/go-unsnap-stream:go_default_library",
        "//vendor/github.com/tinylib/msgp/msgp:go_default_library",
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
)
