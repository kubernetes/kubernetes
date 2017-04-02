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
        "healthcheck.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/renstrom/dedent",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/tools/record",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["healthcheck_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/davecgh/go-spew/spew",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
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
)
