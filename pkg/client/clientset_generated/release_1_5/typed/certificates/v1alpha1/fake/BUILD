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
        "fake_certificates_client.go",
        "fake_certificatesigningrequest.go",
        "fake_certificatesigningrequest_expansion.go",
    ],
    tags = ["automanaged"],
)
