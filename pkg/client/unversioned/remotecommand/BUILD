package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
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
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/transport:go_default_library",
        "//pkg/kubelet/server/remotecommand:go_default_library",
        "//pkg/util/exec:go_default_library",
        "//pkg/util/httpstream:go_default_library",
        "//pkg/util/httpstream/spdy:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/util/term:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "remotecommand_test.go",
        "v2_test.go",
        "v4_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/kubelet/server/remotecommand:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/httpstream:go_default_library",
        "//pkg/util/term:go_default_library",
        "//pkg/util/wait:go_default_library",
    ],
)
