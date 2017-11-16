load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "factory.go",
        "item.go",
        "list_element.go",
        "map_element.go",
        "openapi.go",
        "primitive_element.go",
        "type_element.go",
        "util.go",
        "visitor.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubectl/apply/parse",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubectl/apply:go_default_library",
        "//pkg/kubectl/cmd/util/openapi:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/util/proto:go_default_library",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = ["suite_test.go"],
    data = [
        "//api/openapi-spec:swagger-spec",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubectl/apply/parse_test",
    deps = [
        "//vendor/github.com/onsi/ginkgo:go_default_library",
        "//vendor/github.com/onsi/ginkgo/config:go_default_library",
        "//vendor/github.com/onsi/ginkgo/types:go_default_library",
        "//vendor/github.com/onsi/gomega:go_default_library",
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
