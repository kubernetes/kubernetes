load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "editoptions.go",
        "editor.go",
    ],
    visibility = [
        "//build/visible_to:pkg_kubectl_cmd_util_editor_CONSUMERS",
    ],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/kubectl:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/kubectl/resource:go_default_library",
        "//pkg/kubectl/util/crlf:go_default_library",
        "//pkg/kubectl/util/term:go_default_library",
        "//pkg/printers:go_default_library",
        "//vendor/github.com/evanphx/json-patch:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/mergepatch:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/strategicpatch:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/yaml:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["editor_test.go"],
    library = ":go_default_library",
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
        "//build/visible_to:pkg_kubectl_cmd_util_editor_CONSUMERS",
    ],
)
