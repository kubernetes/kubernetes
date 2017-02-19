package(default_visibility = ["//visibility:public"])

licenses(["notice"])

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
        "set.go",
        "unset.go",
        "use_context.go",
        "view.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/kubectl/cmd/templates:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/printers:go_default_library",
        "//pkg/util/i18n:go_default_library",
        "//vendor:github.com/spf13/cobra",
        "//vendor:k8s.io/apimachinery/pkg/api/meta",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apiserver/pkg/util/flag",
        "//vendor:k8s.io/client-go/tools/clientcmd",
        "//vendor:k8s.io/client-go/tools/clientcmd/api",
        "//vendor:k8s.io/client-go/tools/clientcmd/api/latest",
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
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/kubectl/cmd/util:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/api/equality",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
        "//vendor:k8s.io/apiserver/pkg/util/flag",
        "//vendor:k8s.io/client-go/tools/clientcmd",
        "//vendor:k8s.io/client-go/tools/clientcmd/api",
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
)
