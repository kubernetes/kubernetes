package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "backoff.go",
        "cluster_util.go",
        "configmap.go",
        "delaying_deliverer.go",
        "deployment.go",
        "federated_informer.go",
        "federated_updater.go",
        "handlers.go",
        "meta.go",
        "secret.go",
        "versionize_listoptions.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//federation/apis/federation/v1beta1:go_default_library",
        "//federation/client/clientset_generated/federation_clientset:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/unversioned/clientcmd:go_default_library",
        "//pkg/client/unversioned/clientcmd/api:go_default_library",
        "//pkg/controller/deployment/util:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/flowcontrol:go_default_library",
        "//pkg/util/net:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "delaying_deliverer_test.go",
        "deployment_test.go",
        "federated_informer_test.go",
        "federated_updater_test.go",
        "handlers_test.go",
        "meta_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//federation/apis/federation/v1beta1:go_default_library",
        "//federation/client/clientset_generated/federation_clientset/fake:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/client/clientset_generated/clientset/fake:go_default_library",
        "//pkg/client/testing/core:go_default_library",
        "//pkg/controller/deployment/util:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
    ],
)
