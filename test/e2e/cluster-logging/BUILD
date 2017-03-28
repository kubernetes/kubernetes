package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "es.go",
        "es_utils.go",
        "sd.go",
        "sd_load.go",
        "sd_utils.go",
        "utils.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//vendor:github.com/onsi/ginkgo",
        "//vendor:github.com/onsi/gomega",
        "//vendor:golang.org/x/net/context",
        "//vendor:golang.org/x/oauth2/google",
        "//vendor:google.golang.org/api/logging/v2beta1",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/client-go/util/integer",
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
