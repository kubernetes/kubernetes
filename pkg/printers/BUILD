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
    tags = ["automanaged"],
    deps = [
        "//pkg/util/slice:go_default_library",
        "//vendor/github.com/fatih/camelcase:go_default_library",
        "//vendor/github.com/ghodss/yaml:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/k8s.io/client-go/util/jsonpath:go_default_library",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = ["customcolumn_test.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/printers:go_default_library",
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
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = ["humanreadable_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = ["//vendor/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured:go_default_library"],
)
