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
        "helper.go",
        "set.go",
        "set_image.go",
        "set_resources.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/meta:go_default_library",
        "//pkg/kubectl:go_default_library",
        "//pkg/kubectl/cmd/templates:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/kubectl/resource:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//pkg/util/strategicpatch:go_default_library",
        "//vendor:github.com/spf13/cobra",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["set_image_test.go"],
    data = [
        "//examples:config",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/restclient/fake:go_default_library",
        "//pkg/kubectl/cmd/testing:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/kubectl/resource:go_default_library",
    ],
)
