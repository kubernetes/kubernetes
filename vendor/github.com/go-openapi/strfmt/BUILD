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
    importmap = "k8s.io/kubernetes/vendor/github.com/go-openapi/strfmt",
    importpath = "github.com/go-openapi/strfmt",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/asaskevich/govalidator:go_default_library",
        "//vendor/github.com/globalsign/mgo/bson:go_default_library",
        "//vendor/github.com/go-openapi/errors:go_default_library",
        "//vendor/github.com/mailru/easyjson/jlexer:go_default_library",
        "//vendor/github.com/mailru/easyjson/jwriter:go_default_library",
        "//vendor/github.com/mitchellh/mapstructure:go_default_library",
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
