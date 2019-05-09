package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
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
    importmap = "k8s.io/kubernetes/vendor/k8s.io/code-generator/cmd/go-to-protobuf/protobuf",
    importpath = "k8s.io/code-generator/cmd/go-to-protobuf/protobuf",
    deps = [
        "//staging/src/k8s.io/code-generator/pkg/util:go_default_library",
        "//staging/src/k8s.io/code-generator/third_party/forked/golang/reflect:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/gonum.org/v1/gonum/graph:go_default_library",
        "//vendor/gonum.org/v1/gonum/graph/simple:go_default_library",
        "//vendor/gonum.org/v1/gonum/graph/topo:go_default_library",
        "//vendor/k8s.io/gengo/args:go_default_library",
        "//vendor/k8s.io/gengo/generator:go_default_library",
        "//vendor/k8s.io/gengo/namer:go_default_library",
        "//vendor/k8s.io/gengo/parser:go_default_library",
        "//vendor/k8s.io/gengo/types:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["namer_test.go"],
    embed = [":go_default_library"],
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
