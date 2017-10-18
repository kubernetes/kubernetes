load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "cache.go",
        "container_gc.go",
        "container_reference_manager.go",
        "helpers.go",
        "os.go",
        "pty_unsupported.go",
        "ref.go",
        "resize.go",
        "runtime.go",
        "runtime_cache.go",
        "runtime_cache_fake.go",
        "sync_result.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "pty_linux.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/kubelet/container",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/kubelet/apis/cri/v1alpha1/runtime:go_default_library",
        "//pkg/kubelet/util/format:go_default_library",
        "//pkg/kubelet/util/ioutils:go_default_library",
        "//pkg/util/hash:go_default_library",
        "//pkg/volume:go_default_library",
        "//third_party/forked/golang/expansion:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//vendor/k8s.io/client-go/tools/record:go_default_library",
        "//vendor/k8s.io/client-go/tools/reference:go_default_library",
        "//vendor/k8s.io/client-go/tools/remotecommand:go_default_library",
        "//vendor/k8s.io/client-go/util/flowcontrol:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//vendor/github.com/kr/pty:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = [
        "cache_test.go",
        "helpers_test.go",
        "ref_test.go",
        "sync_result_test.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/container",
    library = ":go_default_library",
    deps = [
        "//pkg/api/install:go_default_library",
        "//pkg/api/legacyscheme:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = ["runtime_cache_test.go"],
    importpath = "k8s.io/kubernetes/pkg/kubelet/container_test",
    deps = [
        ":go_default_library",
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
    visibility = ["//visibility:public"],
)
