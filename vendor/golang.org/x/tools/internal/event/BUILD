load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "event.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/tools/internal/event",
    importpath = "golang.org/x/tools/internal/event",
    visibility = ["//vendor/golang.org/x/tools:__subpackages__"],
    deps = [
        "//vendor/golang.org/x/tools/internal/event/core:go_default_library",
        "//vendor/golang.org/x/tools/internal/event/keys:go_default_library",
        "//vendor/golang.org/x/tools/internal/event/label:go_default_library",
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
        "//vendor/golang.org/x/tools/internal/event/core:all-srcs",
        "//vendor/golang.org/x/tools/internal/event/keys:all-srcs",
        "//vendor/golang.org/x/tools/internal/event/label:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
