load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "conversion.go",
        "defaults.go",
        "doc.go",
        "register.go",
        "zz_generated.conversion.go",
        "zz_generated.deepcopy.go",
        "zz_generated.defaults.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/controller/apis/config/v1alpha1",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/controller/apis/config:go_default_library",
        "//pkg/controller/certificates/signer/config/v1alpha1:go_default_library",
        "//pkg/controller/daemon/config/v1alpha1:go_default_library",
        "//pkg/controller/deployment/config/v1alpha1:go_default_library",
        "//pkg/controller/endpoint/config/v1alpha1:go_default_library",
        "//pkg/controller/endpointslice/config/v1alpha1:go_default_library",
        "//pkg/controller/endpointslicemirroring/config/v1alpha1:go_default_library",
        "//pkg/controller/garbagecollector/config/v1alpha1:go_default_library",
        "//pkg/controller/job/config/v1alpha1:go_default_library",
        "//pkg/controller/namespace/config/v1alpha1:go_default_library",
        "//pkg/controller/nodeipam/config/v1alpha1:go_default_library",
        "//pkg/controller/nodelifecycle/config/v1alpha1:go_default_library",
        "//pkg/controller/podautoscaler/config/v1alpha1:go_default_library",
        "//pkg/controller/podgc/config/v1alpha1:go_default_library",
        "//pkg/controller/replicaset/config/v1alpha1:go_default_library",
        "//pkg/controller/replication/config/v1alpha1:go_default_library",
        "//pkg/controller/resourcequota/config/v1alpha1:go_default_library",
        "//pkg/controller/service/config/v1alpha1:go_default_library",
        "//pkg/controller/serviceaccount/config/v1alpha1:go_default_library",
        "//pkg/controller/statefulset/config/v1alpha1:go_default_library",
        "//pkg/controller/ttlafterfinished/config/v1alpha1:go_default_library",
        "//pkg/controller/volume/attachdetach/config/v1alpha1:go_default_library",
        "//pkg/controller/volume/persistentvolume/config/v1alpha1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/conversion:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/component-base/config/v1alpha1:go_default_library",
        "//staging/src/k8s.io/kube-controller-manager/config/v1alpha1:go_default_library",
        "//vendor/k8s.io/utils/pointer:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["defaults_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/kube-controller-manager/config/v1alpha1:go_default_library",
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
    visibility = ["//visibility:public"],
)
