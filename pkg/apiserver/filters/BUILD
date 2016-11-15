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
        "audit.go",
        "authorization.go",
        "doc.go",
        "errors.go",
        "impersonation.go",
        "requestinfo.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apis/authentication:go_default_library",
        "//pkg/apiserver/request:go_default_library",
        "//pkg/auth/authorizer:go_default_library",
        "//pkg/auth/user:go_default_library",
        "//pkg/httplog:go_default_library",
        "//pkg/serviceaccount:go_default_library",
        "//pkg/util/net:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/pborman/uuid",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "audit_test.go",
        "authorization_test.go",
        "impersonation_test.go",
        "requestinfo_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apis/authentication:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apiserver/request:go_default_library",
        "//pkg/auth/authorizer:go_default_library",
        "//pkg/auth/user:go_default_library",
        "//pkg/util/sets:go_default_library",
    ],
)
