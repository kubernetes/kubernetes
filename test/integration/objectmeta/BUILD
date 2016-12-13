package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["objectmeta_test.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/client/clientset_generated/release_1_5:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//test/integration/framework:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
    ],
)
