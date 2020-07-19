package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "volume_manager.go",
        "volume_manager_fake.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/volumemanager",
    deps = [
        "//pkg/kubelet/config:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/pod:go_default_library",
        "//pkg/kubelet/status:go_default_library",
        "//pkg/kubelet/util/format:go_default_library",
        "//pkg/kubelet/volumemanager/cache:go_default_library",
        "//pkg/kubelet/volumemanager/metrics:go_default_library",
        "//pkg/kubelet/volumemanager/populator:go_default_library",
        "//pkg/kubelet/volumemanager/reconciler:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/csimigration:go_default_library",
        "//pkg/volume/util:go_default_library",
        "//pkg/volume/util/hostutil:go_default_library",
        "//pkg/volume/util/operationexecutor:go_default_library",
        "//pkg/volume/util/types:go_default_library",
        "//pkg/volume/util/volumepathhandler:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        "//staging/src/k8s.io/csi-translation-lib:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
        "//vendor/k8s.io/utils/mount:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["volume_manager_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/kubelet/config:go_default_library",
        "//pkg/kubelet/configmap:go_default_library",
        "//pkg/kubelet/container/testing:go_default_library",
        "//pkg/kubelet/pod:go_default_library",
        "//pkg/kubelet/pod/testing:go_default_library",
        "//pkg/kubelet/secret:go_default_library",
        "//pkg/kubelet/status:go_default_library",
        "//pkg/kubelet/status/testing:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/testing:go_default_library",
        "//pkg/volume/util:go_default_library",
        "//pkg/volume/util/hostutil:go_default_library",
        "//pkg/volume/util/types:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/fake:go_default_library",
        "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        "//staging/src/k8s.io/client-go/util/testing:go_default_library",
        "//vendor/k8s.io/utils/mount:go_default_library",
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
        "//pkg/kubelet/volumemanager/cache:all-srcs",
        "//pkg/kubelet/volumemanager/metrics:all-srcs",
        "//pkg/kubelet/volumemanager/populator:all-srcs",
        "//pkg/kubelet/volumemanager/reconciler:all-srcs",
    ],
    tags = ["automanaged"],
)
