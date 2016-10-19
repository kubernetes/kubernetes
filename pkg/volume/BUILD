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
        "doc.go",
        "metrics_cached.go",
        "metrics_du.go",
        "metrics_errors.go",
        "metrics_nil.go",
        "metrics_statfs.go",
        "plugins.go",
        "util.go",
        "volume.go",
        "volume_linux.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/fields:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/chmod:go_default_library",
        "//pkg/util/chown:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//pkg/util/io:go_default_library",
        "//pkg/util/mount:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/validation:go_default_library",
        "//pkg/volume/util:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "metrics_nil_test.go",
        "plugins_test.go",
        "util_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/watch:go_default_library",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = ["metrics_du_test.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/util/testing:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/testing:go_default_library",
    ],
)
