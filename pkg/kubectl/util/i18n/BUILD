package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["i18n.go"],
    importpath = "k8s.io/kubernetes/pkg/kubectl/util/i18n",
    deps = [
        "//pkg/generated:go_default_library",
        "//vendor/github.com/chai2010/gettext-go/gettext:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["i18n_test.go"],
    importpath = "k8s.io/kubernetes/pkg/kubectl/util/i18n",
    library = ":go_default_library",
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
