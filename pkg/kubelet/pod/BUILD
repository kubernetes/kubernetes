package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "mirror_client.go",
        "pod_manager.go",
    ],
    deps = [
        "//pkg/kubelet/configmap:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/secret:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "mirror_client_test.go",
        "pod_manager_test.go",
    ],
    library = ":go_default_library",
    deps = [
        "//pkg/kubelet/configmap:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/pod/testing:go_default_library",
        "//pkg/kubelet/secret:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
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
        "//pkg/kubelet/pod/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
