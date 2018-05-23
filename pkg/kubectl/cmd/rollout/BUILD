load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "rollout.go",
        "rollout_history.go",
        "rollout_pause.go",
        "rollout_resume.go",
        "rollout_status.go",
        "rollout_undo.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubectl/cmd/rollout",
    visibility = [
        "//build/visible_to:pkg_kubectl_cmd_rollout_CONSUMERS",
    ],
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/kubectl:go_default_library",
        "//pkg/kubectl/cmd/set:go_default_library",
        "//pkg/kubectl/cmd/templates:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/kubectl/genericclioptions:go_default_library",
        "//pkg/kubectl/genericclioptions/printers:go_default_library",
        "//pkg/kubectl/genericclioptions/resource:go_default_library",
        "//pkg/kubectl/polymorphichelpers:go_default_library",
        "//pkg/kubectl/scheme:go_default_library",
        "//pkg/kubectl/util/i18n:go_default_library",
        "//pkg/util/interrupt:go_default_library",
        "//vendor/github.com/renstrom/dedent:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
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
        "//build/visible_to:pkg_kubectl_cmd_rollout_CONSUMERS",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["rollout_pause_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/kubectl/cmd/testing:go_default_library",
        "//pkg/kubectl/genericclioptions:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
        "//vendor/k8s.io/client-go/rest/fake:go_default_library",
    ],
)
