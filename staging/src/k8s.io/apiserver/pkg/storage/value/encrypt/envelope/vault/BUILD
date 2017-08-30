package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "client_test.go",
        "vault_test.go",
    ],
    data = glob(["testdata/**"]),
    importpath = "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/vault",
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor/github.com/hashicorp/vault/api:go_default_library",
        "//vendor/github.com/pborman/uuid:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "client.go",
        "vault.go",
    ],
    importpath = "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/vault",
    tags = ["automanaged"],
    deps = [
        "//vendor/github.com/ghodss/yaml:go_default_library",
        "//vendor/github.com/hashicorp/vault/api:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/value/encrypt/envelope:go_default_library",
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
