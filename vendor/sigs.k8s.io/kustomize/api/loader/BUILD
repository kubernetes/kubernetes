load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "fileloader.go",
        "loader.go",
        "loadrestrictions.go",
        "loadrestrictions_string.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/api/loader",
    importpath = "sigs.k8s.io/kustomize/api/loader",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/filesys:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/git:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/ifc:go_default_library",
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
