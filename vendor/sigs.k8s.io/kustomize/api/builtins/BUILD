load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "annotationstransformer.go",
        "configmapgenerator.go",
        "doc.go",
        "hashtransformer.go",
        "imagetagtransformer.go",
        "inventorytransformer.go",
        "labeltransformer.go",
        "legacyordertransformer.go",
        "namespacetransformer.go",
        "patchjson6902transformer.go",
        "patchstrategicmergetransformer.go",
        "patchtransformer.go",
        "prefixsuffixtransformer.go",
        "replicacounttransformer.go",
        "secretgenerator.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/api/builtins",
    importpath = "sigs.k8s.io/kustomize/api/builtins",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/evanphx/json-patch:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/hasher:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/ifc:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/inventory:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/kv:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/resid:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/resmap:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/resource:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/transform:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/types:go_default_library",
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
