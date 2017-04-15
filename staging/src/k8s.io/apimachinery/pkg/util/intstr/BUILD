package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["intstr_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = ["//vendor/github.com/ghodss/yaml:go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "generated.pb.go",
        "intstr.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/github.com/go-openapi/spec:go_default_library",
        "//vendor/github.com/gogo/protobuf/proto:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/google/gofuzz:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/openapi:go_default_library",
    ],
)
