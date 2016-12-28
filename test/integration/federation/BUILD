package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["server_test.go"],
    tags = ["automanaged"],
    deps = [
        "//federation/apis/federation/v1beta1:go_default_library",
        "//federation/cmd/federation-apiserver/app:go_default_library",
        "//federation/cmd/federation-apiserver/app/options:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/batch/v1:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
    ],
)
