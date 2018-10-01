load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "0doc.go",
        "binc.go",
        "cbor.go",
        "decode.go",
        "encode.go",
        "fast-path.generated.go",
        "gen.generated.go",
        "gen-helper.generated.go",
        "goversion_arrayof_gte_go15.go",
        "goversion_arrayof_lt_go15.go",
        "goversion_makemap_gte_go19.go",
        "goversion_makemap_lt_go19.go",
        "goversion_unsupported_lt_go14.go",
        "goversion_vendor_eq_go15.go",
        "goversion_vendor_eq_go16.go",
        "goversion_vendor_gte_go17.go",
        "goversion_vendor_lt_go15.go",
        "helper.go",
        "helper_internal.go",
        "helper_not_unsafe.go",
        "helper_unsafe.go",
        "json.go",
        "msgpack.go",
        "rpc.go",
        "simple.go",
        "time.go",
        "z.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/ugorji/go/codec",
    importpath = "github.com/ugorji/go/codec",
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
