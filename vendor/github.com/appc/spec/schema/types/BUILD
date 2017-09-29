load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "acidentifier.go",
        "ackind.go",
        "acname.go",
        "annotations.go",
        "app.go",
        "date.go",
        "dependencies.go",
        "doc.go",
        "environment.go",
        "errors.go",
        "event_handler.go",
        "exec.go",
        "hash.go",
        "isolator.go",
        "isolator_linux_specific.go",
        "isolator_resources.go",
        "isolator_unix.go",
        "labels.go",
        "mountpoint.go",
        "port.go",
        "semver.go",
        "url.go",
        "user_annotations.go",
        "user_labels.go",
        "uuid.go",
        "volume.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/appc/spec/schema/common:go_default_library",
        "//vendor/github.com/appc/spec/schema/types/resource:go_default_library",
        "//vendor/github.com/coreos/go-semver/semver:go_default_library",
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
        "//vendor/github.com/appc/spec/schema/types/resource:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
