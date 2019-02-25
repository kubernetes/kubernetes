load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "auth_basic.go",
        "http.go",
        "opts.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/mesos/mesos-go/api/v1/lib/httpcli",
    importpath = "github.com/mesos/mesos-go/api/v1/lib/httpcli",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/mesos/mesos-go/api/v1/lib:go_default_library",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/client:go_default_library",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/debug:go_default_library",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/encoding:go_default_library",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/encoding/codecs:go_default_library",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/encoding/framing:go_default_library",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/httpcli/apierrors:go_default_library",
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/recordio:go_default_library",
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
        "//vendor/github.com/mesos/mesos-go/api/v1/lib/httpcli/apierrors:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
