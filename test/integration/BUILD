package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "utils.go",
    ],
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/core/v1:go_default_library",
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
        "//test/integration/apiserver:all-srcs",
        "//test/integration/auth:all-srcs",
        "//test/integration/client:all-srcs",
        "//test/integration/configmap:all-srcs",
        "//test/integration/defaulttolerationseconds:all-srcs",
        "//test/integration/deployment:all-srcs",
        "//test/integration/etcd:all-srcs",
        "//test/integration/evictions:all-srcs",
        "//test/integration/examples:all-srcs",
        "//test/integration/framework:all-srcs",
        "//test/integration/garbagecollector:all-srcs",
        "//test/integration/kubectl:all-srcs",
        "//test/integration/master:all-srcs",
        "//test/integration/metrics:all-srcs",
        "//test/integration/objectmeta:all-srcs",
        "//test/integration/openshift:all-srcs",
        "//test/integration/pods:all-srcs",
        "//test/integration/quota:all-srcs",
        "//test/integration/replicaset:all-srcs",
        "//test/integration/replicationcontroller:all-srcs",
        "//test/integration/scheduler:all-srcs",
        "//test/integration/scheduler_perf:all-srcs",
        "//test/integration/secrets:all-srcs",
        "//test/integration/serviceaccount:all-srcs",
        "//test/integration/storageclasses:all-srcs",
        "//test/integration/ttlcontroller:all-srcs",
        "//test/integration/volume:all-srcs",
    ],
    tags = ["automanaged"],
)
