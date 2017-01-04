package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["x509_test.go"],
    data = glob(["testdata/*"]),
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/client-go/pkg/util/sets",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "x509.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/client-go/pkg/util/errors",
        "//vendor:k8s.io/client-go/pkg/util/sets",
    ],
)
