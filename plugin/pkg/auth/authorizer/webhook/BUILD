package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["webhook.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/authorization/install:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
        "//vendor:k8s.io/apiserver/pkg/util/cache",
        "//vendor:k8s.io/apiserver/pkg/webhook",
        "//vendor:k8s.io/client-go/kubernetes/typed/authorization/v1beta1",
        "//vendor:k8s.io/client-go/pkg/apis/authorization/install",
        "//vendor:k8s.io/client-go/pkg/apis/authorization/v1beta1",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "certs_test.go",
        "webhook_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/authorization/v1beta1:go_default_library",
        "//pkg/client/unversioned/clientcmd/api/v1:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
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
