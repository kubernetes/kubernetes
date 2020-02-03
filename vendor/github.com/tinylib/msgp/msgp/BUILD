load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "advise_linux.go",
        "advise_other.go",
        "circular.go",
        "defs.go",
        "edit.go",
        "elsize.go",
        "errors.go",
        "extension.go",
        "file.go",
        "file_port.go",
        "integers.go",
        "json.go",
        "json_bytes.go",
        "number.go",
        "read.go",
        "read_bytes.go",
        "size.go",
        "unsafe.go",
        "write.go",
        "write_bytes.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/tinylib/msgp/msgp",
    importpath = "github.com/tinylib/msgp/msgp",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/philhofer/fwd:go_default_library"],
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
