package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["desired_state_of_world_populator.go"],
    importpath = "k8s.io/kubernetes/cmd/kubelet/pkg/volumemanager/populator",
    deps = [
        "//cmd/kubelet/pkg/config:go_default_library",
        "//cmd/kubelet/pkg/container:go_default_library",
        "//cmd/kubelet/pkg/pod:go_default_library",
        "//cmd/kubelet/pkg/status:go_default_library",
        "//cmd/kubelet/pkg/util/format:go_default_library",
        "//cmd/kubelet/pkg/volumemanager/cache:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/util:go_default_library",
        "//pkg/volume/util/types:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
    embed = [":go_default_library"],
    deps = [
        "//cmd/kubelet/pkg/configmap:go_default_library",
        "//cmd/kubelet/pkg/container/testing:go_default_library",
        "//cmd/kubelet/pkg/pod:go_default_library",
        "//cmd/kubelet/pkg/pod/testing:go_default_library",
        "//cmd/kubelet/pkg/secret:go_default_library",
        "//cmd/kubelet/pkg/status:go_default_library",
        "//cmd/kubelet/pkg/status/testing:go_default_library",
        "//cmd/kubelet/pkg/volumemanager/cache:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/volume/testing:go_default_library",
        "//pkg/volume/util:go_default_library",
        "//pkg/volume/util/types:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/fake:go_default_library",
        "//staging/src/k8s.io/client-go/testing:go_default_library",
        "//staging/src/k8s.io/component-base/featuregate/testing:go_default_library",
    ],
)
