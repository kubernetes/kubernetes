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
        "benchmark_util.go",
        "container.go",
        "doc.go",
        "image_list.go",
        "resource_collector.go",
        "simple_mount.go",
        "util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/componentconfig:go_default_library",
        "//pkg/apis/componentconfig/v1alpha1:go_default_library",
        "//pkg/kubelet/api/v1alpha1/stats:go_default_library",
        "//pkg/util/procfs:go_default_library",
        "//test/e2e/common:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//test/e2e/perftype:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "apparmor_test.go",
        "cgroup_manager_test.go",
        "container_manager_test.go",
        "density_test.go",
        "disk_eviction_test.go",
        "dynamic_kubelet_configuration_test.go",
        "e2e_node_suite_test.go",
        "garbage_collector_test.go",
        "image_id_test.go",
        "inode_eviction_test.go",
        "kubelet_test.go",
        "lifecycle_hook_test.go",
        "log_path_test.go",
        "memory_eviction_test.go",
        "mirror_pod_test.go",
        "resource_usage_test.go",
        "restart_test.go",
        "runtime_conformance_test.go",
        "summary_test.go",
        "volume_manager_test.go",
    ],
    library = ":go_default_library",
    tags = [
        "automanaged",
        "integration",
    ],
    deps = [
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/kubelet:go_default_library",
        "//pkg/kubelet/api/v1alpha1/stats:go_default_library",
        "//pkg/kubelet/cm:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/dockertools:go_default_library",
        "//pkg/kubelet/images:go_default_library",
        "//pkg/kubelet/metrics:go_default_library",
        "//pkg/metrics:go_default_library",
        "//pkg/security/apparmor:go_default_library",
        "//test/e2e/common:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//test/e2e_node/services:go_default_library",
        "//test/e2e_node/system:go_default_library",
        "//test/utils:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
    ],
)

genrule(
    name = "gen_e2e_node.test",
    testonly = 1,
    srcs = [":go_default_test"],
    outs = ["e2e_node.test"],
    cmd = "srcs=($(SRCS)); cp $$(dirname $${srcs[0]})/go_default_test $@;",
    output_to_bindir = 1,
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
        "//test/e2e_node/builder:all-srcs",
        "//test/e2e_node/environment:all-srcs",
        "//test/e2e_node/remote:all-srcs",
        "//test/e2e_node/runner/local:all-srcs",
        "//test/e2e_node/runner/remote:all-srcs",
        "//test/e2e_node/services:all-srcs",
        "//test/e2e_node/system:all-srcs",
    ],
    tags = ["automanaged"],
)
