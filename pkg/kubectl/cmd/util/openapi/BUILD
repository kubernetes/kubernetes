package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "extensions.go",
        "openapi.go",
        "openapi_getter.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi",
    deps = [
        "//vendor/github.com/go-openapi/spec:go_default_library",
        "//vendor/github.com/googleapis/gnostic/OpenAPIv2:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/client-go/discovery:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/util/proto:go_default_library",
    ],
)

go_test(
    name = "go_default_xtest",
    size = "small",
    srcs = [
        "openapi_getter_test.go",
        "openapi_suite_test.go",
        "openapi_test.go",
    ],
    data = ["//api/openapi-spec:swagger-spec"],
    importpath = "k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi_test",
    deps = [
        ":go_default_library",
        "//pkg/kubectl/cmd/util/openapi/testing:go_default_library",
        "//vendor/github.com/onsi/ginkgo:go_default_library",
        "//vendor/github.com/onsi/ginkgo/config:go_default_library",
        "//vendor/github.com/onsi/ginkgo/types:go_default_library",
        "//vendor/github.com/onsi/gomega:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//pkg/kubectl/cmd/util/openapi/testing:all-srcs",
        "//pkg/kubectl/cmd/util/openapi/validation:all-srcs",
    ],
    tags = ["automanaged"],
)

filegroup(
    name = "testdata",
    srcs = glob(["testdata/*"]),
)
