load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "openapi.go",
        "validation.go",
    ],
    importpath = "k8s.io/apimachinery/pkg/util/openapi",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/googleapis/gnostic/OpenAPIv2:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/json:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/yaml:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/util/proto:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/util/proto/validation:go_default_library",
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
        "//staging/src/k8s.io/apimachinery/pkg/util/openapi/testing:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_xtest",
    srcs = [
        "openapi_suite_test.go",
        "openapi_test.go",
    ],
    data = ["//api/openapi-spec:swagger-spec"],
    importpath = "k8s.io/apimachinery/pkg/util/openapi_test",
    deps = [
        "//vendor/github.com/onsi/ginkgo:go_default_library",
        "//vendor/github.com/onsi/ginkgo/config:go_default_library",
        "//vendor/github.com/onsi/ginkgo/types:go_default_library",
        "//vendor/github.com/onsi/gomega:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/openapi/testing:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/util/proto:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["validation_test.go"],
    importpath = "k8s.io/apimachinery/pkg/util/openapi",
    library = ":go_default_library",
    deps = [
        "//vendor/github.com/onsi/ginkgo:go_default_library",
        "//vendor/github.com/onsi/gomega:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/openapi/testing:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/util/proto/validation:go_default_library",
    ],
)
