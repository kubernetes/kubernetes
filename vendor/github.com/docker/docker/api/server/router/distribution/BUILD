load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "backend.go",
        "distribution.go",
        "distribution_routes.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/api/server/router/distribution",
    importpath = "github.com/docker/docker/api/server/router/distribution",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/docker/distribution:go_default_library",
        "//vendor/github.com/docker/distribution/manifest/manifestlist:go_default_library",
        "//vendor/github.com/docker/distribution/manifest/schema1:go_default_library",
        "//vendor/github.com/docker/distribution/manifest/schema2:go_default_library",
        "//vendor/github.com/docker/distribution/reference:go_default_library",
        "//vendor/github.com/docker/docker/api/server/httputils:go_default_library",
        "//vendor/github.com/docker/docker/api/server/router:go_default_library",
        "//vendor/github.com/docker/docker/api/types:go_default_library",
        "//vendor/github.com/docker/docker/api/types/registry:go_default_library",
        "//vendor/github.com/opencontainers/image-spec/specs-go/v1:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
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
