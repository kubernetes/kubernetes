load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bson.go",
        "date.go",
        "default.go",
        "doc.go",
        "duration.go",
        "format.go",
        "time.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kube-openapi/pkg/validation/strfmt",
    importpath = "k8s.io/kube-openapi/pkg/validation/strfmt",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/asaskevich/govalidator:go_default_library",
        "//vendor/github.com/mitchellh/mapstructure:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/validation/errors:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/validation/strfmt/bson:go_default_library",
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
        "//vendor/k8s.io/kube-openapi/pkg/validation/strfmt/bson:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
