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
        "auth.go",
        "doc.go",
        "server.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/api/v1/validation:go_default_library",
        "//pkg/kubelet/cm:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/server/portforward:go_default_library",
        "//pkg/kubelet/server/remotecommand:go_default_library",
        "//pkg/kubelet/server/stats:go_default_library",
        "//pkg/kubelet/server/streaming:go_default_library",
        "//pkg/util/configz:go_default_library",
        "//pkg/util/limitwriter:go_default_library",
        "//pkg/util/term:go_default_library",
        "//pkg/volume:go_default_library",
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/info/v2",
        "//vendor:github.com/prometheus/client_golang/prometheus",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apiserver/pkg/authentication/authenticator",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
        "//vendor:k8s.io/apiserver/pkg/healthz",
        "//vendor:k8s.io/apiserver/pkg/httplog",
        "//vendor:k8s.io/apiserver/pkg/util/flushwriter",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "auth_test.go",
        "server_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/kubelet/cm:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/container/testing:go_default_library",
        "//pkg/kubelet/server/remotecommand:go_default_library",
        "//pkg/kubelet/server/stats:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//pkg/util/httpstream:go_default_library",
        "//pkg/util/httpstream/spdy:go_default_library",
        "//pkg/util/term:go_default_library",
        "//pkg/util/testing:go_default_library",
        "//pkg/volume:go_default_library",
        "//vendor:github.com/google/cadvisor/info/v1",
        "//vendor:github.com/google/cadvisor/info/v2",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:github.com/stretchr/testify/require",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
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
        "//pkg/kubelet/server/portforward:all-srcs",
        "//pkg/kubelet/server/remotecommand:all-srcs",
        "//pkg/kubelet/server/stats:all-srcs",
        "//pkg/kubelet/server/streaming:all-srcs",
    ],
    tags = ["automanaged"],
)
