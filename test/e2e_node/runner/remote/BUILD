package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "remote",
    library = ":go_default_library",
    tags = ["automanaged"],
)

go_library(
    name = "go_default_library",
    srcs = ["run_remote.go"],
    tags = ["automanaged"],
    deps = [
        "//test/e2e_node/remote:go_default_library",
        "//vendor:github.com/ghodss/yaml",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/pborman/uuid",
        "//vendor:golang.org/x/oauth2",
        "//vendor:golang.org/x/oauth2/google",
        "//vendor:google.golang.org/api/compute/v1",
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
