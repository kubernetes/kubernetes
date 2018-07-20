load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "middleware.go",
        "router_swapper.go",
        "server.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/api/server",
    importpath = "github.com/docker/docker/api/server",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/docker/docker/api/server/httputils:go_default_library",
        "//vendor/github.com/docker/docker/api/server/middleware:go_default_library",
        "//vendor/github.com/docker/docker/api/server/router:go_default_library",
        "//vendor/github.com/docker/docker/api/server/router/debug:go_default_library",
        "//vendor/github.com/docker/docker/dockerversion:go_default_library",
        "//vendor/github.com/gorilla/mux:go_default_library",
        "//vendor/github.com/sirupsen/logrus:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/docker/docker/api/server/backend/build:all-srcs",
        "//vendor/github.com/docker/docker/api/server/httputils:all-srcs",
        "//vendor/github.com/docker/docker/api/server/middleware:all-srcs",
        "//vendor/github.com/docker/docker/api/server/router:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
