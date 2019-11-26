load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "decode.go",
        "encode.go",
        "expfmt.go",
        "text_create.go",
        "text_parse.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/prometheus/common/expfmt",
    importpath = "github.com/prometheus/common/expfmt",
    visibility = ["//staging/src/k8s.io/component-base/metrics:prometheus_import_allow_list"],
    deps = [
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
        "//vendor/github.com/matttproud/golang_protobuf_extensions/pbutil:go_default_library",
        "//vendor/github.com/prometheus/client_model/go:go_default_library",
        "//vendor/github.com/prometheus/common/internal/bitbucket.org/ww/goautoneg:go_default_library",
        "//vendor/github.com/prometheus/common/model:go_default_library",
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
