package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "api.go",
        "controller.go",
        "crudtester.go",
        "daemonset.go",
        "federation.go",
        "secret.go",
        "util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//federation/apis/federation/v1beta1:go_default_library",
        "//federation/client/clientset_generated/federation_clientset:go_default_library",
        "//federation/cmd/federation-apiserver/app:go_default_library",
        "//federation/cmd/federation-apiserver/app/options:go_default_library",
        "//federation/pkg/federation-controller/cluster:go_default_library",
        "//federation/pkg/federation-controller/daemonset:go_default_library",
        "//federation/pkg/federation-controller/secret:go_default_library",
        "//federation/pkg/typeadapters:go_default_library",
        "//federation/pkg/typeadapters/crudtester:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/master:go_default_library",
        "//test/integration/framework:go_default_library",
        "//vendor:github.com/pborman/uuid",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/client-go/rest",
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
