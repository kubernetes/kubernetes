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
        "//pkg/api/unversioned:go_default_library",
        "//pkg/client/record:go_default_library",
        "//pkg/kubelet/api/v1alpha1/runtime:go_default_library",
        "//pkg/kubelet/util/format:go_default_library",
        "//pkg/kubelet/util/ioutils:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//pkg/util/flowcontrol:go_default_library",
        "//pkg/util/hash:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/util/term:go_default_library",
        "//pkg/volume:go_default_library",
        "//third_party/forked/golang/expansion:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/kr/pty",
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
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/types:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
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
