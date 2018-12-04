load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["controllermanager.go"],
    importpath = "k8s.io/kubernetes/cmd/cloud-controller-manager/app",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/cloud-controller-manager/app/config:go_default_library",
        "//cmd/cloud-controller-manager/app/options:go_default_library",
        "//cmd/controller-manager/app:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/controller:go_default_library",
        "//pkg/controller/cloud:go_default_library",
        "//pkg/controller/route:go_default_library",
        "//pkg/controller/service:go_default_library",
        "//pkg/util/configz:go_default_library",
        "//pkg/util/flag:go_default_library",
        "//pkg/version/verflag:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/server/healthz:go_default_library",
        "//vendor/k8s.io/client-go/informers:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
        "//vendor/k8s.io/client-go/tools/leaderelection:go_default_library",
        "//vendor/k8s.io/client-go/tools/leaderelection/resourcelock:go_default_library",
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
        "//cmd/cloud-controller-manager/app/config:all-srcs",
        "//cmd/cloud-controller-manager/app/options:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
