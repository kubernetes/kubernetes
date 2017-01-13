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
        "//pkg/api/v1:go_default_library",
        "//pkg/client/record:go_default_library",
        "//pkg/kubelet/api/v1alpha1/runtime:go_default_library",
        "//pkg/kubelet/util/format:go_default_library",
        "//pkg/kubelet/util/ioutils:go_default_library",
        "//pkg/util/flowcontrol:go_default_library",
        "//pkg/util/hash:go_default_library",
        "//pkg/util/term:go_default_library",
        "//pkg/volume:go_default_library",
        "//third_party/forked/golang/expansion:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/kr/pty",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
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
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/types",
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
