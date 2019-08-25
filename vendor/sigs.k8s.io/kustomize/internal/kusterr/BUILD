load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "configmaperror.go",
        "patcherror.go",
        "resourceerror.go",
        "secreterror.go",
        "yamlformaterror.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/internal/kusterr",
    importpath = "sigs.k8s.io/kustomize/internal/kusterr",
    visibility = ["//vendor/sigs.k8s.io/kustomize:__subpackages__"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "configmaperror_test.go",
        "patcherror_test.go",
        "resourceerror_test.go",
        "secreterror_test.go",
        "yamlformaterror_test.go",
    ],
    embed = [":go_default_library"],
    tags = ["manual"],
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
