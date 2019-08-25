load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["SecretsFromDatabase.go"],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/plugin/someteam.example.com/v1/secretsfromdatabase",
    importpath = "sigs.k8s.io/kustomize/plugin/someteam.example.com/v1/secretsfromdatabase",
    visibility = ["//visibility:private"],
    deps = [
        "//vendor/sigs.k8s.io/kustomize/pkg/ifc:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/resmap:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/types:go_default_library",
        "//vendor/sigs.k8s.io/yaml:go_default_library",
    ],
)

go_binary(
    name = "secretsfromdatabase",
    embed = [":go_default_library"],
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["SecretsFromDatabase_test.go"],
    embed = [":go_default_library"],
    tags = ["manual"],
    deps = [
        "//vendor/sigs.k8s.io/kustomize/pkg/kusttest:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/plugin:go_default_library",
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
