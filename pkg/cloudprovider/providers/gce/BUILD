package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "gce.go",
        "token_source.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/api/v1/service:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/util/net/sets:go_default_library",
        "//pkg/volume:go_default_library",
        "//vendor:cloud.google.com/go/compute/metadata",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/prometheus/client_golang/prometheus",
        "//vendor:golang.org/x/oauth2",
        "//vendor:golang.org/x/oauth2/google",
        "//vendor:google.golang.org/api/compute/v1",
        "//vendor:google.golang.org/api/container/v1",
        "//vendor:google.golang.org/api/googleapi",
        "//vendor:gopkg.in/gcfg.v1",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/client-go/util/flowcontrol",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["gce_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
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
)
