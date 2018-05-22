load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

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

go_library(
    name = "go_default_library",
    srcs = [
        "get.go",
        "get_flags.go",
        "humanreadable_flags.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubectl/cmd/get",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/apis/core:go_default_library",
        "//pkg/kubectl:go_default_library",
        "//pkg/kubectl/cmd/templates:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/kubectl/cmd/util/openapi:go_default_library",
        "//pkg/kubectl/genericclioptions:go_default_library",
        "//pkg/kubectl/genericclioptions/resource:go_default_library",
        "//pkg/kubectl/scheme:go_default_library",
        "//pkg/kubectl/util/i18n:go_default_library",
        "//pkg/printers:go_default_library",
        "//pkg/printers/internalversion:go_default_library",
        "//pkg/util/interrupt:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1beta1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "get_test.go",
        "humanreadable_flags_test.go",
    ],
    data = [
        "//api/openapi-spec:swagger-spec",
        "//test/e2e/testing-manifests:all-srcs",
        "//test/fixtures",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/api/testing:go_default_library",
        "//pkg/apis/core:go_default_library",
        "//pkg/apis/core/v1:go_default_library",
        "//pkg/kubectl/cmd/testing:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/kubectl/cmd/util/openapi:go_default_library",
        "//pkg/kubectl/cmd/util/openapi/testing:go_default_library",
        "//pkg/kubectl/genericclioptions:go_default_library",
        "//pkg/kubectl/genericclioptions/resource:go_default_library",
        "//pkg/kubectl/scheme:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer/json:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer/streaming:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
        "//vendor/k8s.io/client-go/rest/fake:go_default_library",
        "//vendor/k8s.io/client-go/rest/watch:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/util/proto:go_default_library",
    ],
)
