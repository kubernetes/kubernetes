load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "envfortest.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/plugin",
    importpath = "sigs.k8s.io/kustomize/plugin",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/sigs.k8s.io/kustomize/pkg/pgmconfig:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/plugins:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/sigs.k8s.io/kustomize/plugin/builtin:all-srcs",
        "//vendor/sigs.k8s.io/kustomize/plugin/someteam.example.com/v1/bashedconfigmap:all-srcs",
        "//vendor/sigs.k8s.io/kustomize/plugin/someteam.example.com/v1/dateprefixer:all-srcs",
        "//vendor/sigs.k8s.io/kustomize/plugin/someteam.example.com/v1/printworkdir:all-srcs",
        "//vendor/sigs.k8s.io/kustomize/plugin/someteam.example.com/v1/secretsfromdatabase:all-srcs",
        "//vendor/sigs.k8s.io/kustomize/plugin/someteam.example.com/v1/sedtransformer:all-srcs",
        "//vendor/sigs.k8s.io/kustomize/plugin/someteam.example.com/v1/someservicegenerator:all-srcs",
        "//vendor/sigs.k8s.io/kustomize/plugin/someteam.example.com/v1/stringprefixer:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
