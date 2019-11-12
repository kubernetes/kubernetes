load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "configmapargs.go",
        "doc.go",
        "erronlybuiltinpluginsallowed.go",
        "errunabletofind.go",
        "fieldspec.go",
        "fix.go",
        "garbagepolicy.go",
        "garbagepolicy_string.go",
        "genargs.go",
        "generationbehavior.go",
        "generatorargs.go",
        "generatoroptions.go",
        "image.go",
        "inventory.go",
        "kustomization.go",
        "kvpairsources.go",
        "loadrestrictions.go",
        "loadrestrictions_string.go",
        "objectmeta.go",
        "pair.go",
        "patch.go",
        "patchjson6902.go",
        "patchstrategicmerge.go",
        "patchtarget.go",
        "pluginconfig.go",
        "pluginrestrictions.go",
        "pluginrestrictions_string.go",
        "replica.go",
        "secretargs.go",
        "selector.go",
        "typemeta.go",
        "var.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/api/types",
    importpath = "sigs.k8s.io/kustomize/api/types",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/resid:go_default_library",
        "//vendor/sigs.k8s.io/yaml:go_default_library",
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
