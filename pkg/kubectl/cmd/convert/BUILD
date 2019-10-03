load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "convert.go",
        "import_known_versions.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubectl/cmd/convert",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/cli-runtime/pkg/genericclioptions:go_default_library",
        "//staging/src/k8s.io/cli-runtime/pkg/printers:go_default_library",
        "//staging/src/k8s.io/cli-runtime/pkg/resource:go_default_library",
        "//staging/src/k8s.io/internal-api/api/legacyscheme:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/apps/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/authentication/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/authorization/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/autoscaling/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/batch/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/certificates/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/coordination/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/core:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/core/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/events/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/extensions/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/policy/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/rbac/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/scheduling/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/settings/install:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/storage/install:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/cmd/util:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/util/i18n:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/util/templates:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/validation:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["convert_test.go"],
    data = [
        "//test/fixtures",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/cli-runtime/pkg/genericclioptions:go_default_library",
        "//staging/src/k8s.io/client-go/rest/fake:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/cmd/testing:go_default_library",
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
