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
        "//pkg/apis/extensions:go_default_library",
        "//pkg/registry/extensions/thirdpartyresourcedata:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
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
        "//pkg/api:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
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
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apiserver/pkg/registry/generic",
        "//vendor:k8s.io/apiserver/pkg/registry/rest",
        "//vendor:k8s.io/apiserver/pkg/server",
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
