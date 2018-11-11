package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "errors.go",
        "request_cache.go",
        "server.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/server/streaming",
    deps = [
        "//pkg/kubelet/apis/cri/runtime/v1alpha2:go_default_library",
        "//pkg/kubelet/server/portforward:go_default_library",
        "//pkg/kubelet/server/remotecommand:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/clock:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/remotecommand:go_default_library",
        "//staging/src/k8s.io/client-go/tools/remotecommand:go_default_library",
        "//vendor/github.com/emicklei/go-restful:go_default_library",
        "//vendor/google.golang.org/grpc:go_default_library",
        "//vendor/google.golang.org/grpc/codes:go_default_library",
        "//vendor/google.golang.org/grpc/status:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "request_cache_test.go",
        "server_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/apis/core:go_default_library",
        "//pkg/kubelet/apis/cri/runtime/v1alpha2:go_default_library",
        "//pkg/kubelet/server/portforward:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/clock:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/client-go/tools/remotecommand:go_default_library",
        "//staging/src/k8s.io/client-go/transport/spdy:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/github.com/stretchr/testify/require:go_default_library",
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
