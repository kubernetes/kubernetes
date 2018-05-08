package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "cacher_test.go",
        "parser_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//vendor/github.com/go-openapi/spec:go_default_library",
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
        "//vendor/github.com/googleapis/gnostic/OpenAPIv2:go_default_library",
        "//vendor/github.com/googleapis/gnostic/compiler:go_default_library",
        "//vendor/gopkg.in/yaml.v2:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/util/proto:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "cacher.go",
        "interfaces.go",
        "parser.go",
        "resource.go",
    ],
    importpath = "k8s.io/apimachinery/pkg/runtime/openapi",
    deps = [
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
        "//vendor/github.com/googleapis/gnostic/OpenAPIv2:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/util/proto:go_default_library",
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
)
