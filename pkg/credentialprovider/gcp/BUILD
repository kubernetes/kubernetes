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
        "jwt.go",
        "metadata.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/credentialprovider:go_default_library",
        "//pkg/util/net:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/spf13/pflag",
        "//vendor:golang.org/x/oauth2",
        "//vendor:golang.org/x/oauth2/google",
        "//vendor:golang.org/x/oauth2/jwt",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "jwt_test.go",
        "metadata_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/credentialprovider:go_default_library",
        "//pkg/util/net:go_default_library",
    ],
)
