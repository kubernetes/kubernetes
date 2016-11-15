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
    srcs = ["gcp.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/client/restclient:go_default_library",
        "//pkg/util/jsonpath:go_default_library",
        "//pkg/util/yaml:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/context",
        "//vendor:golang.org/x/oauth2",
        "//vendor:golang.org/x/oauth2/google",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["gcp_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = ["//vendor:golang.org/x/oauth2"],
)
