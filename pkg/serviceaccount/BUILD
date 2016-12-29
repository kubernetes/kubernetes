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
        "jwt.go",
        "util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/api/validation:go_default_library",
        "//pkg/auth/authenticator:go_default_library",
        "//pkg/auth/user:go_default_library",
        "//vendor:github.com/dgrijalva/jwt-go",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["util_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
)

go_test(
    name = "go_default_xtest",
    srcs = ["jwt_test.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/client/clientset_generated/clientset/fake:go_default_library",
        "//pkg/controller/serviceaccount:go_default_library",
        "//pkg/serviceaccount:go_default_library",
    ],
)
