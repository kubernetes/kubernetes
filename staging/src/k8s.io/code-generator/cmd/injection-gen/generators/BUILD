load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "generator_client.go",
        "generator_client_fake.go",
        "generator_factory.go",
        "generator_factory_fake.go",
        "generator_informer.go",
        "generator_informer_fake.go",
        "namesystems.go",
        "packages.go",
        "tags.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/code-generator/cmd/injection-gen/generators",
    importpath = "k8s.io/code-generator/cmd/injection-gen/generators",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/code-generator/cmd/client-gen/generators/util:go_default_library",
        "//staging/src/k8s.io/code-generator/cmd/client-gen/types:go_default_library",
        "//staging/src/k8s.io/code-generator/cmd/injection-gen/args:go_default_library",
        "//staging/src/k8s.io/code-generator/pkg/namer:go_default_library",
        "//vendor/k8s.io/gengo/args:go_default_library",
        "//vendor/k8s.io/gengo/generator:go_default_library",
        "//vendor/k8s.io/gengo/namer:go_default_library",
        "//vendor/k8s.io/gengo/types:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
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
