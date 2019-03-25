load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["edit.go"],
    importpath = "k8s.io/kubernetes/pkg/kubectl/cmd/edit",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/kubectl/cmd/util/editor:go_default_library",
        "//pkg/kubectl/util/i18n:go_default_library",
        "//pkg/kubectl/util/templates:go_default_library",
        "//staging/src/k8s.io/cli-runtime/pkg/genericclioptions:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["edit_test.go"],
    data = [
        "//pkg/kubectl/cmd/edit/testdata:all-srcs",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/kubectl/cmd/apply:go_default_library",
        "//pkg/kubectl/cmd/create:go_default_library",
        "//pkg/kubectl/cmd/testing:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/cli-runtime/pkg/genericclioptions:go_default_library",
        "//staging/src/k8s.io/cli-runtime/pkg/resource:go_default_library",
        "//staging/src/k8s.io/client-go/rest/fake:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/gopkg.in/yaml.v2:go_default_library",
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
        "//pkg/kubectl/cmd/edit/testdata:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
