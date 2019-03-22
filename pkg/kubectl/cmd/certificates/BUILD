load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["certificates.go"],
    importpath = "k8s.io/kubernetes/pkg/kubectl/cmd/certificates",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/kubectl/scheme:go_default_library",
        "//pkg/kubectl/util/i18n:go_default_library",
        "//pkg/kubectl/util/templates:go_default_library",
        "//staging/src/k8s.io/api/certificates/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/cli-runtime/pkg/genericclioptions:go_default_library",
        "//staging/src/k8s.io/cli-runtime/pkg/printers:go_default_library",
        "//staging/src/k8s.io/cli-runtime/pkg/resource:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/typed/certificates/v1beta1:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
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
