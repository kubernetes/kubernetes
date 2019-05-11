package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    size = "small",
    srcs = ["oidc_test.go"],
    data = glob(["testdata/**"]),
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apiserver/pkg/authentication/authenticator:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//vendor/github.com/coreos/go-oidc:go_default_library",
        "//vendor/gopkg.in/square/go-jose.v2:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = ["oidc.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/plugin/pkg/authenticator/token/oidc",
    importpath = "k8s.io/apiserver/plugin/pkg/authenticator/token/oidc",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/net:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/authenticator:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//staging/src/k8s.io/client-go/util/cert:go_default_library",
        "//vendor/github.com/coreos/go-oidc:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
