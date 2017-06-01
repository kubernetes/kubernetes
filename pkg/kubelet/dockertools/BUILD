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
        "docker.go",
        "fake_docker_client.go",
        "instrumented_docker.go",
        "kube_docker_client.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/credentialprovider:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/images:go_default_library",
        "//pkg/kubelet/metrics:go_default_library",
        "//vendor/github.com/docker/distribution/digest:go_default_library",
        "//vendor/github.com/docker/distribution/reference:go_default_library",
        "//vendor/github.com/docker/docker/pkg/jsonmessage:go_default_library",
        "//vendor/github.com/docker/docker/pkg/stdcopy:go_default_library",
        "//vendor/github.com/docker/engine-api/client:go_default_library",
        "//vendor/github.com/docker/engine-api/types:go_default_library",
        "//vendor/github.com/docker/engine-api/types/container:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/k8s.io/client-go/util/clock:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "docker_test.go",
        "kube_docker_client_test.go",
    ],
    library = ":go_default_library",
    tags = [
        "automanaged",
    ],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/credentialprovider:go_default_library",
        "//pkg/kubelet/images:go_default_library",
        "//pkg/util/hash:go_default_library",
        "//vendor/github.com/docker/docker/pkg/jsonmessage:go_default_library",
        "//vendor/github.com/docker/engine-api/types:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
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
