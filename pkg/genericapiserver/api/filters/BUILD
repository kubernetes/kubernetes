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
        "audit.go",
        "authorization.go",
        "doc.go",
        "impersonation.go",
        "requestinfo.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apis/authentication:go_default_library",
        "//pkg/genericapiserver/api/handlers/responsewriters:go_default_library",
        "//pkg/serviceaccount:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/pborman/uuid",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
        "//vendor:k8s.io/apiserver/pkg/httplog",
        "//vendor:k8s.io/apiserver/pkg/request",
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
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/authentication:go_default_library",
        "//pkg/apis/batch:go_default_library",
        "//pkg/genericapiserver/api/handlers/responsewriters:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
        "//vendor:k8s.io/apiserver/pkg/request",
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
