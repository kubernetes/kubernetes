load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["importers.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/golang/dep/internal/importers",
    importpath = "github.com/golang/dep/internal/importers",
    visibility = ["//vendor/github.com/golang/dep:__subpackages__"],
    deps = [
        "//vendor/github.com/golang/dep:go_default_library",
        "//vendor/github.com/golang/dep/gps:go_default_library",
        "//vendor/github.com/golang/dep/internal/importers/glide:go_default_library",
        "//vendor/github.com/golang/dep/internal/importers/glock:go_default_library",
        "//vendor/github.com/golang/dep/internal/importers/godep:go_default_library",
        "//vendor/github.com/golang/dep/internal/importers/govend:go_default_library",
        "//vendor/github.com/golang/dep/internal/importers/govendor:go_default_library",
        "//vendor/github.com/golang/dep/internal/importers/gvt:go_default_library",
        "//vendor/github.com/golang/dep/internal/importers/vndr:go_default_library",
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
        "//vendor/github.com/golang/dep/internal/importers/base:all-srcs",
        "//vendor/github.com/golang/dep/internal/importers/glide:all-srcs",
        "//vendor/github.com/golang/dep/internal/importers/glock:all-srcs",
        "//vendor/github.com/golang/dep/internal/importers/godep:all-srcs",
        "//vendor/github.com/golang/dep/internal/importers/govend:all-srcs",
        "//vendor/github.com/golang/dep/internal/importers/govendor:all-srcs",
        "//vendor/github.com/golang/dep/internal/importers/gvt:all-srcs",
        "//vendor/github.com/golang/dep/internal/importers/vndr:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
