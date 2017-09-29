load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "any.go",
        "doc.go",
        "duration.go",
        "timestamp.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
        "//vendor/github.com/golang/protobuf/ptypes/any:go_default_library",
        "//vendor/github.com/golang/protobuf/ptypes/duration:go_default_library",
        "//vendor/github.com/golang/protobuf/ptypes/timestamp:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/golang/protobuf/ptypes/any:all-srcs",
        "//vendor/github.com/golang/protobuf/ptypes/duration:all-srcs",
        "//vendor/github.com/golang/protobuf/ptypes/timestamp:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
