load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "controllerrevision.go",
        "daemonset.go",
        "daemonsetcondition.go",
        "daemonsetspec.go",
        "daemonsetstatus.go",
        "daemonsetupdatestrategy.go",
        "deployment.go",
        "deploymentcondition.go",
        "deploymentspec.go",
        "deploymentstatus.go",
        "deploymentstrategy.go",
        "replicaset.go",
        "replicasetcondition.go",
        "replicasetspec.go",
        "replicasetstatus.go",
        "rollingupdatedaemonset.go",
        "rollingupdatedeployment.go",
        "rollingupdatestatefulsetstrategy.go",
        "statefulset.go",
        "statefulsetcondition.go",
        "statefulsetspec.go",
        "statefulsetstatus.go",
        "statefulsetupdatestrategy.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/client-go/applyconfigurations/apps/v1",
    importpath = "k8s.io/client-go/applyconfigurations/apps/v1",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/apps/v1:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
        "//staging/src/k8s.io/client-go/applyconfigurations/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/applyconfigurations/meta/v1:go_default_library",
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
