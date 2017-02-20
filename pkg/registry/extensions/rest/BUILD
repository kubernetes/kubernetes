package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["thirdparty_controller_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/registry/extensions/thirdpartyresourcedata:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "storage_extensions.go",
        "thirdparty_controller.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion:go_default_library",
        "//pkg/registry/extensions/controller/storage:go_default_library",
        "//pkg/registry/extensions/daemonset/storage:go_default_library",
        "//pkg/registry/extensions/deployment/storage:go_default_library",
        "//pkg/registry/extensions/ingress/storage:go_default_library",
        "//pkg/registry/extensions/networkpolicy/storage:go_default_library",
        "//pkg/registry/extensions/podsecuritypolicy/storage:go_default_library",
        "//pkg/registry/extensions/replicaset/storage:go_default_library",
        "//pkg/registry/extensions/thirdpartyresource/storage:go_default_library",
        "//pkg/registry/extensions/thirdpartyresourcedata:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
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
