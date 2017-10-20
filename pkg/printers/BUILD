package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "common.go",
        "customcolumn.go",
        "humanreadable.go",
        "interface.go",
        "json.go",
        "jsonpath.go",
        "name.go",
        "printers.go",
        "tabwriter.go",
        "template.go",
        "versioned.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/printers",
    deps = [
        "//vendor/github.com/ghodss/yaml:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1alpha1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//vendor/k8s.io/client-go/util/jsonpath:go_default_library",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = ["customcolumn_test.go"],
    importpath = "k8s.io/kubernetes/pkg/printers_test",
    deps = [
        ":go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/legacyscheme:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
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
        "//pkg/printers/internalversion:all-srcs",
        "//pkg/printers/storage:all-srcs",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = ["humanreadable_test.go"],
    importpath = "k8s.io/kubernetes/pkg/printers",
    library = ":go_default_library",
    deps = [
        "//pkg/api:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1alpha1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
    ],
)
