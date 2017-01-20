package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["gcp.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/client/restclient:go_default_library",
        "//pkg/util/jsonpath:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/context",
        "//vendor:golang.org/x/oauth2",
        "//vendor:golang.org/x/oauth2/google",
        "//vendor:k8s.io/apimachinery/pkg/util/yaml",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["gcp_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/oauth2"],
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
