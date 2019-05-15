package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["checkpoint_manager.go"],
    importpath = "k8s.io/kubernetes/cmd/kubelet/pkg/checkpointmanager",
    deps = [
        "//cmd/kubelet/pkg/checkpointmanager/errors:go_default_library",
        "//cmd/kubelet/pkg/util/store:go_default_library",
        "//pkg/util/filesystem:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["checkpoint_manager_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//cmd/kubelet/pkg/checkpointmanager/checksum:go_default_library",
        "//cmd/kubelet/pkg/checkpointmanager/testing:go_default_library",
        "//cmd/kubelet/pkg/checkpointmanager/testing/example_checkpoint_formats/v1:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
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
        "//cmd/kubelet/pkg/checkpointmanager/checksum:all-srcs",
        "//cmd/kubelet/pkg/checkpointmanager/errors:all-srcs",
        "//cmd/kubelet/pkg/checkpointmanager/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
