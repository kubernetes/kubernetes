package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "config.go",
        "doc.go",
        "keyring.go",
        "plugins.go",
        "provider.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/credentialprovider",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/github.com/docker/docker/api/types:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "config_test.go",
        "keyring_test.go",
        "provider_test.go",
    ],
    embed = [":go_default_library"],
    deps = ["//vendor/github.com/docker/docker/api/types:go_default_library"],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//pkg/credentialprovider/aws:all-srcs",
        "//pkg/credentialprovider/azure:all-srcs",
        "//pkg/credentialprovider/gcp:all-srcs",
        "//pkg/credentialprovider/rancher:all-srcs",
        "//pkg/credentialprovider/secrets:all-srcs",
    ],
    tags = ["automanaged"],
)
