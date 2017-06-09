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
        "cache.go",
        "container_gc.go",
        "container_reference_manager.go",
        "helpers.go",
        "os.go",
        "pty_linux.go",
        "ref.go",
        "resize.go",
        "runtime.go",
        "runtime_cache.go",
        "runtime_cache_fake.go",
        "sync_result.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/api/v1/ref:go_default_library",
        "//pkg/kubelet/apis/cri/v1alpha1/runtime:go_default_library",
        "//pkg/kubelet/events:go_default_library",
        "//pkg/kubelet/util/format:go_default_library",
        "//pkg/kubelet/util/ioutils:go_default_library",
        "//pkg/util/hash:go_default_library",
        "//pkg/volume:go_default_library",
        "//third_party/forked/golang/expansion:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/kr/pty:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//vendor/k8s.io/client-go/pkg/api/v1:go_default_library",
        "//vendor/k8s.io/client-go/tools/record:go_default_library",
        "//vendor/k8s.io/client-go/tools/remotecommand:go_default_library",
        "//vendor/k8s.io/client-go/util/flowcontrol:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "cache_test.go",
        "helpers_test.go",
        "ref_test.go",
        "sync_result_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = ["runtime_cache_test.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/container/testing:go_default_library",
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
        "//pkg/kubelet/container/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
