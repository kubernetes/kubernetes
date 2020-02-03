load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "arraycontainer.go",
        "arraycontainer_gen.go",
        "bitmapcontainer.go",
        "bitmapcontainer_gen.go",
        "byte_input.go",
        "clz.go",
        "clz_compat.go",
        "ctz.go",
        "ctz_compat.go",
        "fastaggregation.go",
        "manyiterator.go",
        "parallel.go",
        "popcnt.go",
        "popcnt_amd64.s",
        "popcnt_asm.go",
        "popcnt_compat.go",
        "popcnt_generic.go",
        "popcnt_slices.go",
        "priorityqueue.go",
        "roaring.go",
        "roaringarray.go",
        "roaringarray_gen.go",
        "runcontainer.go",
        "runcontainer_gen.go",
        "serialization.go",
        "serialization_generic.go",
        "serialization_littleendian.go",
        "setutil.go",
        "shortiterator.go",
        "util.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/RoaringBitmap/roaring",
    importpath = "github.com/RoaringBitmap/roaring",
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
)
