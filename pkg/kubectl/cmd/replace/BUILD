load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["replace.go"],
    importpath = "k8s.io/kubernetes/pkg/kubectl/cmd/replace",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubectl:go_default_library",
        "//pkg/kubectl/cmd/delete:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/kubectl/scheme:go_default_library",
        "//pkg/kubectl/util/i18n:go_default_library",
        "//pkg/kubectl/util/templates:go_default_library",
        "//pkg/kubectl/validation:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/cli-runtime/pkg/genericclioptions:go_default_library",
        "//staging/src/k8s.io/cli-runtime/pkg/resource:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["replace_test.go"],
    data = [
        "//test/e2e/testing-manifests:all-srcs",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/kubectl/cmd/testing:go_default_library",
        "//pkg/kubectl/scheme:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/cli-runtime/pkg/genericclioptions:go_default_library",
        "//staging/src/k8s.io/cli-runtime/pkg/resource:go_default_library",
        "//staging/src/k8s.io/client-go/rest/fake:go_default_library",
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
