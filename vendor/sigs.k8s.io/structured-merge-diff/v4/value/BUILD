load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "allocator.go",
        "doc.go",
        "fields.go",
        "jsontagutil.go",
        "list.go",
        "listreflect.go",
        "listunstructured.go",
        "map.go",
        "mapreflect.go",
        "mapunstructured.go",
        "reflectcache.go",
        "scalar.go",
        "structreflect.go",
        "value.go",
        "valuereflect.go",
        "valueunstructured.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/structured-merge-diff/v4/value",
    importpath = "sigs.k8s.io/structured-merge-diff/v4/value",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/json-iterator/go:go_default_library",
        "//vendor/gopkg.in/yaml.v2:go_default_library",
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
