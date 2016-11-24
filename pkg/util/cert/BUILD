package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "cert.go",
        "csr.go",
        "io.go",
        "pem.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/certificates:go_default_library",
        "//pkg/apis/certificates/v1alpha1:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["csr_test.go"],
    data = [
        "testdata/dontUseThisKey.pem",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
