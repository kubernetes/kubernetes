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
        "errors.go",
        "request_cache.go",
        "server.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/kubelet/api/v1alpha1/runtime:go_default_library",
        "//pkg/kubelet/server/portforward:go_default_library",
        "//pkg/kubelet/server/remotecommand:go_default_library",
        "//pkg/util/clock:go_default_library",
        "//pkg/util/term:go_default_library",
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:google.golang.org/grpc",
        "//vendor:google.golang.org/grpc/codes",
        "//vendor:k8s.io/apimachinery/pkg/types",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "request_cache_test.go",
        "server_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/unversioned/remotecommand:go_default_library",
        "//pkg/kubelet/api/v1alpha1/runtime:go_default_library",
        "//pkg/kubelet/server/portforward:go_default_library",
        "//pkg/kubelet/server/remotecommand:go_default_library",
        "//pkg/util/term:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:github.com/stretchr/testify/require",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/util/clock",
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
