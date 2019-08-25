load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "addbase.go",
        "addmetadata.go",
        "addpatch.go",
        "addresource.go",
        "all.go",
        "configmap.go",
        "flagsandargs.go",
        "secret.go",
        "util.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/pkg/commands/edit/add",
    importpath = "sigs.k8s.io/kustomize/pkg/commands/edit/add",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/commands/kustfile:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/fs:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/ifc:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/patch:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/pgmconfig:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/types:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "addbase_test.go",
        "addmetadata_test.go",
        "addpatch_test.go",
        "addresource_test.go",
        "configmap_test.go",
        "flagsandargs_test.go",
        "secret_test.go",
    ],
    embed = [":go_default_library"],
    tags = ["manual"],
    deps = [
        "//vendor/sigs.k8s.io/kustomize/pkg/commands/kustfile:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/fs:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/loader:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/types:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/validators:go_default_library",
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
