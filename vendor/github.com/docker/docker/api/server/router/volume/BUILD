load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "backend.go",
        "volume.go",
        "volume_routes.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/api/server/router/volume",
    importpath = "github.com/docker/docker/api/server/router/volume",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/docker/docker/api/server/httputils:go_default_library",
        "//vendor/github.com/docker/docker/api/server/router:go_default_library",
        "//vendor/github.com/docker/docker/api/types:go_default_library",
        "//vendor/github.com/docker/docker/api/types/filters:go_default_library",
        "//vendor/github.com/docker/docker/api/types/volume:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
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
    visibility = ["//visibility:public"],
)
