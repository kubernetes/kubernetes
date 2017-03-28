package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "bootstrapsigner_test.go",
        "common_test.go",
        "jws_test.go",
        "tokencleaner_test.go",
        "util_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/bootstrap/api:go_default_library",
        "//vendor:github.com/davecgh/go-spew/spew",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/client-go/kubernetes/fake",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/testing",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "bootstrapsigner.go",
        "doc.go",
        "jws.go",
        "tokencleaner.go",
        "util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/bootstrap/api:go_default_library",
        "//pkg/util/metrics:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/square/go-jose",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/kubernetes",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/tools/cache",
        "//vendor:k8s.io/client-go/util/workqueue",
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
