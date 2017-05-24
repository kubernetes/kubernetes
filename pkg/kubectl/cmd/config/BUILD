package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
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
        "set.go",
        "unset.go",
        "use_context.go",
        "view.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/client/unversioned/clientcmd:go_default_library",
        "//pkg/client/unversioned/clientcmd/api:go_default_library",
        "//pkg/client/unversioned/clientcmd/api/latest:go_default_library",
        "//pkg/kubectl:go_default_library",
        "//pkg/kubectl/cmd/templates:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//pkg/util/flag:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//vendor:github.com/spf13/cobra",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "config_test.go",
        "create_authinfo_test.go",
        "current_context_test.go",
        "delete_cluster_test.go",
        "delete_context_test.go",
        "get_clusters_test.go",
        "get_contexts_test.go",
        "navigation_step_parser_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/client/unversioned/clientcmd:go_default_library",
        "//pkg/client/unversioned/clientcmd/api:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/util/diff:go_default_library",
        "//pkg/util/flag:go_default_library",
    ],
)
