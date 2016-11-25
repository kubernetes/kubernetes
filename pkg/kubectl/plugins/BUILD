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
        "loader.go",
        "plugins.go",
        "runner.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/ghodss/yaml",
        "//vendor:github.com/golang/glog",
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

go_test(
    name = "go_default_test",
    srcs = [
        "loader_test.go",
        "plugins_test.go",
        "runner_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
)
