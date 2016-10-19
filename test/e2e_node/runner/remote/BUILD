package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_binary(
    name = "remote",
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
