load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "external.go",
        "golist.go",
        "golist_overlay.go",
        "packages.go",
        "visit.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/tools/go/packages",
    importpath = "golang.org/x/tools/go/packages",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/tools/go/gcexportdata:go_default_library",
        "//vendor/golang.org/x/tools/go/internal/packagesdriver:go_default_library",
        "//vendor/golang.org/x/tools/internal/gopathwalk:go_default_library",
        "//vendor/golang.org/x/tools/internal/semver:go_default_library",
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
