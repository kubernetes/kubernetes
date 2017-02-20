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
        "garbagecollector.go",
        "metrics.go",
        "rate_limiter_helper.go",
        "uid_cache.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/controller/garbagecollector/metaonly:go_default_library",
        "//pkg/util/metrics:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["garbagecollector_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/controller/garbagecollector/metaonly:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
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
        "//pkg/controller/garbagecollector/metaonly:all-srcs",
    ],
    tags = ["automanaged"],
)
