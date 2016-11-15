package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "cmd.go",
        "generator.go",
        "import_tracker.go",
        "namer.go",
        "package.go",
        "parser.go",
        "tags.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//third_party/forked/golang/reflect:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/spf13/pflag",
        "//vendor:k8s.io/gengo/args",
        "//vendor:k8s.io/gengo/generator",
        "//vendor:k8s.io/gengo/namer",
        "//vendor:k8s.io/gengo/parser",
        "//vendor:k8s.io/gengo/types",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["namer_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
