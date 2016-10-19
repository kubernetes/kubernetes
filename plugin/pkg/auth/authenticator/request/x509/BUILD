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
        "doc.go",
        "x509.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/auth/user:go_default_library",
        "//pkg/util/errors:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["x509_test.go"],
    data = [
        "testdata/client.config.json",
        "testdata/client.csr.json",
        "testdata/client-expired.pem",
        "testdata/client-valid.pem",
        "testdata/generate.sh",
        "testdata/intermediate.config.json",
        "testdata/intermediate.csr.json",
        "testdata/intermediate.pem",
        "testdata/root.csr.json",
        "testdata/root.pem",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = ["//pkg/auth/user:go_default_library"],
)
