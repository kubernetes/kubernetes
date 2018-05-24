load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "attachdetachcontroller.go",
        "cloudprovider.go",
        "csrsigningcontroller.go",
        "daemonsetcontroller.go",
        "debugging.go",
        "deploymentcontroller.go",
        "deprecatedcontroller.go",
        "endpointcontroller.go",
        "garbagecollectorcontroller.go",
        "generic.go",
        "hpacontroller.go",
        "insecure_serving.go",
        "jobcontroller.go",
        "kubecloudshared.go",
        "namespacecontroller.go",
        "nodeipamcontroller.go",
        "nodelifecyclecontroller.go",
        "persistentvolumebindercontroller.go",
        "podgccontroller.go",
        "replicasetcontroller.go",
        "replicationcontroller.go",
        "resourcequotacontroller.go",
        "serviceaccountcontroller.go",
        "servicecontroller.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/controller-manager/app/options",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/controller-manager/app:go_default_library",
        "//pkg/apis/componentconfig:go_default_library",
        "//pkg/client/leaderelectionconfig:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/server/options:go_default_library",
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
