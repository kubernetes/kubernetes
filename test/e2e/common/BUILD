package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "configmap.go",
        "container_probe.go",
        "docker_containers.go",
        "downward_api.go",
        "downwardapi_volume.go",
        "empty_dir.go",
        "expansion.go",
        "host_path.go",
        "init_container.go",
        "kubelet_etc_hosts.go",
        "networking.go",
        "pods.go",
        "privileged.go",
        "secrets.go",
        "sysctl.go",
        "util.go",
        "volumes.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/unversioned:go_default_library",
        "//pkg/kubelet:go_default_library",
        "//pkg/kubelet/events:go_default_library",
        "//pkg/kubelet/sysctl:go_default_library",
        "//pkg/labels:go_default_library",
        "//pkg/util/intstr:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/uuid:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//pkg/watch:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//test/utils:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/onsi/ginkgo",
        "//vendor:github.com/onsi/gomega",
        "//vendor:golang.org/x/net/websocket",
    ],
)
