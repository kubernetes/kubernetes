load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "helper.go",
        "set.go",
        "set_env.go",
        "set_image.go",
        "set_resources.go",
        "set_selector.go",
        "set_serviceaccount.go",
        "set_subject.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubectl/cmd/set",
    visibility = ["//build/visible_to:pkg_kubectl_cmd_set_CONSUMERS"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/apis/rbac:go_default_library",
        "//pkg/kubectl:go_default_library",
        "//pkg/kubectl/cmd/templates:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/kubectl/cmd/util/env:go_default_library",
        "//pkg/kubectl/resource:go_default_library",
        "//pkg/kubectl/util/i18n:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/strategicpatch:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/validation:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "set_env_test.go",
        "set_image_test.go",
        "set_resources_test.go",
        "set_selector_test.go",
        "set_serviceaccount_test.go",
        "set_subject_test.go",
        "set_test.go",
    ],
    data = [
        "//examples:config",
        "//test/fixtures",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubectl/cmd/set",
    library = ":go_default_library",
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/apis/apps:go_default_library",
        "//pkg/apis/batch:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/rbac:go_default_library",
        "//pkg/kubectl/cmd/testing:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/kubectl/resource:go_default_library",
        "//pkg/printers:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
        "//vendor/k8s.io/client-go/rest/fake:go_default_library",
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
        "//build/visible_to:pkg_kubectl_cmd_set_CONSUMERS",
    ],
)
