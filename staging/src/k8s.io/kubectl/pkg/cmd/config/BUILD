load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "config.go",
        "create_authinfo.go",
        "create_cluster.go",
        "create_context.go",
        "current_context.go",
        "delete_cluster.go",
        "delete_context.go",
        "get_clusters.go",
        "get_contexts.go",
        "navigation_step_parser.go",
        "rename_context.go",
        "set.go",
        "unset.go",
        "use_context.go",
        "view.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kubectl/pkg/cmd/config",
    importpath = "k8s.io/kubectl/pkg/cmd/config",
    visibility = [
        "//build/visible_to:pkg_kubectl_cmd_config_CONSUMERS",
    ],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/cli-runtime/pkg/genericclioptions:go_default_library",
        "//staging/src/k8s.io/cli-runtime/pkg/printers:go_default_library",
        "//staging/src/k8s.io/client-go/tools/clientcmd:go_default_library",
        "//staging/src/k8s.io/client-go/tools/clientcmd/api:go_default_library",
        "//staging/src/k8s.io/client-go/tools/clientcmd/api/latest:go_default_library",
        "//staging/src/k8s.io/component-base/cli/flag:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/cmd/util:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/scheme:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/util/i18n:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/util/printers:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/util/templates:go_default_library",
        "//vendor/github.com/liggitt/tabwriter:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "config_test.go",
        "create_authinfo_test.go",
        "create_cluster_test.go",
        "create_context_test.go",
        "current_context_test.go",
        "delete_cluster_test.go",
        "delete_context_test.go",
        "get_clusters_test.go",
        "get_contexts_test.go",
        "navigation_step_parser_test.go",
        "rename_context_test.go",
        "set_test.go",
        "unset_test.go",
        "use_context_test.go",
        "view_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//staging/src/k8s.io/cli-runtime/pkg/genericclioptions:go_default_library",
        "//staging/src/k8s.io/client-go/tools/clientcmd:go_default_library",
        "//staging/src/k8s.io/client-go/tools/clientcmd/api:go_default_library",
        "//staging/src/k8s.io/component-base/cli/flag:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/cmd/util:go_default_library",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = [
        "//build/visible_to:pkg_kubectl_cmd_config_CONSUMERS",
    ],
)
