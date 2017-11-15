load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "x509.go",
    ],
    importpath = "k8s.io/apiserver/pkg/authentication/request/x509",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/authentication/authenticator:go_default_library",
        "//pkg/authentication/user:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["x509_test.go"],
    data = glob(["testdata/**"]),
    importpath = "k8s.io/apiserver/pkg/authentication/request/x509",
    library = ":go_default_library",
    deps = [
        "//pkg/authentication/authenticator:go_default_library",
        "//pkg/authentication/user:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
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
    visibility = ["//visibility:public"],
)
