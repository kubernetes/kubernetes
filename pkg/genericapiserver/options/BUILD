package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "authentication.go",
        "authorization.go",
        "doc.go",
        "etcd.go",
        "server_run_options.go",
        "serving.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/admission:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/genericapiserver/authenticator:go_default_library",
        "//pkg/genericapiserver/authorizer:go_default_library",
        "//pkg/storage/storagebackend:go_default_library",
        "//pkg/util/cert:go_default_library",
        "//pkg/util/config:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/spf13/pflag",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/client-go/kubernetes/typed/authentication/v1beta1",
        "//vendor:k8s.io/client-go/kubernetes/typed/authorization/v1beta1",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/tools/clientcmd",
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
