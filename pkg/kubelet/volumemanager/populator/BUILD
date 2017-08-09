package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["desired_state_of_world_populator.go"],
    deps = [
        "//pkg/kubelet/config:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/pod:go_default_library",
        "//pkg/kubelet/status:go_default_library",
        "//pkg/kubelet/util/format:go_default_library",
        "//pkg/kubelet/volumemanager/cache:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/util/types:go_default_library",
        "//pkg/volume/util/volumehelper:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/scheme:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = ["desired_state_of_world_populator_test.go"],
    library = ":go_default_library",
    deps = [
        "//pkg/kubelet/configmap:go_default_library",
        "//pkg/kubelet/container/testing:go_default_library",
        "//pkg/kubelet/pod:go_default_library",
        "//pkg/kubelet/pod/testing:go_default_library",
        "//pkg/kubelet/secret:go_default_library",
        "//pkg/kubelet/status:go_default_library",
        "//pkg/kubelet/status/testing:go_default_library",
        "//pkg/kubelet/volumemanager/cache:go_default_library",
        "//pkg/volume/testing:go_default_library",
        "//pkg/volume/util/types:go_default_library",
        "//pkg/volume/util/volumehelper:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/fake:go_default_library",
    ],
)
