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
        "resize.go",
        "resizeevents.go",
        "setsize.go",
        "term.go",
        "term_writer.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/util/interrupt:go_default_library",
        "//vendor/github.com/docker/docker/pkg/term:go_default_library",
        "//vendor/github.com/mitchellh/go-wordwrap:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//vendor/k8s.io/client-go/tools/remotecommand:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["term_writer_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
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
