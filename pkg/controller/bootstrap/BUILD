package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "bootstrapsigner_test.go",
        "common_test.go",
        "jws_test.go",
        "tokencleaner_test.go",
        "util_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/clientset_generated/clientset/fake:go_default_library",
        "//pkg/client/testing/core:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//vendor:github.com/davecgh/go-spew/spew",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "bootstrapsigner.go",
        "doc.go",
        "jws.go",
        "tokencleaner.go",
        "util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/fields:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/metrics:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/square/go-jose",
    ],
)
