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
        "//vendor:github.com/cloudflare/cfssl/config",
        "//vendor:github.com/cloudflare/cfssl/csr",
        "//vendor:github.com/cloudflare/cfssl/helpers",
        "//vendor:github.com/cloudflare/cfssl/initca",
        "//vendor:github.com/cloudflare/cfssl/signer",
        "//vendor:github.com/cloudflare/cfssl/signer/local",
        "//vendor:github.com/golang/glog",
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
