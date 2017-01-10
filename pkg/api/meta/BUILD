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
        "doc.go",
        "errors.go",
        "firsthit_restmapper.go",
        "help.go",
        "interfaces.go",
        "meta.go",
        "multirestmapper.go",
        "priority.go",
        "restmapper.go",
        "unstructured.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/apis/meta/v1/unstructured:go_default_library",
        "//pkg/conversion:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "multirestmapper_test.go",
        "priority_test.go",
        "restmapper_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
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
        "//pkg/api/meta/metatypes:all-srcs",
    ],
    tags = ["automanaged"],
)
