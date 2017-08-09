package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "fake_client_test.go",
        "listwatch_test.go",
        "portfoward_test.go",
        "remotecommand_test.go",
    ],
    library = ":go_default_library",
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/install:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/fake:go_default_library",
        "//pkg/kubelet/server/portforward:go_default_library",
        "//pkg/kubelet/server/remotecommand:go_default_library",
        "//vendor/github.com/stretchr/testify/require:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/httpstream:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/remotecommand:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/k8s.io/client-go/tools/portforward:go_default_library",
        "//vendor/k8s.io/client-go/tools/remotecommand:go_default_library",
        "//vendor/k8s.io/client-go/transport/spdy:go_default_library",
        "//vendor/k8s.io/client-go/util/testing:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = ["doc.go"],
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
