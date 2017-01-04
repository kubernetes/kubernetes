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
        "context.go",
        "doc.go",
        "requestcontext.go",
        "requestinfo.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/types:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:golang.org/x/net/context",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["requestinfo_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = ["//pkg/util/sets:go_default_library"],
)

go_test(
    name = "go_default_xtest",
    srcs = ["context_test.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/genericapiserver/api/request:go_default_library",
        "//pkg/types:go_default_library",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)
