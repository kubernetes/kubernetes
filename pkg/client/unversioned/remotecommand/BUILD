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
        "doc.go",
        "errorstream.go",
        "remotecommand.go",
        "v1.go",
        "v2.go",
        "v3.go",
        "v4.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/kubelet/server/remotecommand:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//pkg/util/term:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/util/httpstream",
        "//vendor:k8s.io/apimachinery/pkg/util/httpstream/spdy",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/transport",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "remotecommand_test.go",
        "v2_test.go",
        "v4_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/kubelet/server/remotecommand:go_default_library",
        "//pkg/util/term:go_default_library",
        "//vendor:github.com/stretchr/testify/require",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/httpstream",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/client-go/rest",
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
