load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "getentries.go",
        "logclient.go",
        "multilog.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/google/certificate-transparency-go/client",
    importpath = "github.com/google/certificate-transparency-go/client",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
        "//vendor/github.com/golang/protobuf/ptypes:go_default_library",
        "//vendor/github.com/google/certificate-transparency-go:go_default_library",
        "//vendor/github.com/google/certificate-transparency-go/client/configpb:go_default_library",
        "//vendor/github.com/google/certificate-transparency-go/jsonclient:go_default_library",
        "//vendor/github.com/google/certificate-transparency-go/tls:go_default_library",
        "//vendor/github.com/google/certificate-transparency-go/x509:go_default_library",
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
        "//vendor/github.com/google/certificate-transparency-go/client/configpb:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
