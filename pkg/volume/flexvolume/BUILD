package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "attacher.go",
        "attacher-defaults.go",
        "detacher.go",
        "detacher-defaults.go",
        "driver-call.go",
        "expander.go",
        "expander-defaults.go",
        "fake_watcher.go",
        "mounter.go",
        "mounter-defaults.go",
        "plugin.go",
        "plugin-defaults.go",
        "probe.go",
        "unmounter.go",
        "unmounter-defaults.go",
        "util.go",
        "volume.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/volume/flexvolume",
    deps = [
        "//pkg/util/filesystem:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/util:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/github.com/fsnotify/fsnotify:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
        "//vendor/k8s.io/utils/exec:go_default_library",
        "//vendor/k8s.io/utils/mount:go_default_library",
        "//vendor/k8s.io/utils/strings:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "attacher_test.go",
        "common_test.go",
        "detacher_test.go",
        "driver-call_test.go",
        "flexvolume_test.go",
        "mounter_test.go",
        "plugin_test.go",
        "probe_test.go",
        "unmounter_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/util/filesystem:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/volume/testing:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/client-go/util/testing:go_default_library",
        "//test/utils/harness:go_default_library",
        "//vendor/github.com/fsnotify/fsnotify:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/k8s.io/utils/exec:go_default_library",
        "//vendor/k8s.io/utils/exec/testing:go_default_library",
        "//vendor/k8s.io/utils/mount:go_default_library",
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
