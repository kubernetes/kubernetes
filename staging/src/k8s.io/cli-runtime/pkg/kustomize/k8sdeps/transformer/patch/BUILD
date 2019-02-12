load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "patch.go",
        "patchconflictdetector.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/cli-runtime/pkg/kustomize/k8sdeps/transformer/patch",
    importpath = "k8s.io/cli-runtime/pkg/kustomize/k8sdeps/transformer/patch",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/mergepatch:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/strategicpatch:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/scheme:go_default_library",
        "//vendor/github.com/evanphx/json-patch:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/gvk:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/resmap:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/resource:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/transformers:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["patch_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/cli-runtime/pkg/kustomize/k8sdeps/kunstruct:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/gvk:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/resid:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/resmap:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/resource:go_default_library",
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
