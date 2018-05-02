package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "fake_kuberuntime_manager.go",
        "helpers.go",
        "instrumented_services.go",
        "kuberuntime_container.go",
        "kuberuntime_gc.go",
        "kuberuntime_image.go",
        "kuberuntime_logs.go",
        "kuberuntime_manager.go",
        "kuberuntime_sandbox.go",
        "labels.go",
        "legacy.go",
        "security_context.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "helpers_unsupported.go",
            "kuberuntime_container_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "helpers_unsupported.go",
            "kuberuntime_container_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "helpers_unsupported.go",
            "kuberuntime_container_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "helpers_unsupported.go",
            "kuberuntime_container_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "helpers_linux.go",
            "kuberuntime_container_linux.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "helpers_unsupported.go",
            "kuberuntime_container_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "helpers_unsupported.go",
            "kuberuntime_container_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "helpers_unsupported.go",
            "kuberuntime_container_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "helpers_unsupported.go",
            "kuberuntime_container_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "helpers_unsupported.go",
            "kuberuntime_container_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "helpers_windows.go",
            "kuberuntime_container_windows.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "k8s.io/kubernetes/pkg/kubelet/kuberuntime",
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/apis/core/v1/helper:go_default_library",
        "//pkg/credentialprovider:go_default_library",
        "//pkg/credentialprovider/secrets:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/kubelet/apis/cri:go_default_library",
        "//pkg/kubelet/apis/cri/runtime/v1alpha2:go_default_library",
        "//pkg/kubelet/cm:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/events:go_default_library",
        "//pkg/kubelet/images:go_default_library",
        "//pkg/kubelet/kuberuntime/logs:go_default_library",
        "//pkg/kubelet/lifecycle:go_default_library",
        "//pkg/kubelet/metrics:go_default_library",
        "//pkg/kubelet/prober/results:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//pkg/kubelet/util/cache:go_default_library",
        "//pkg/kubelet/util/format:go_default_library",
        "//pkg/security/apparmor:go_default_library",
        "//pkg/securitycontext:go_default_library",
        "//pkg/util/parsers:go_default_library",
        "//pkg/util/selinux:go_default_library",
        "//pkg/util/tail:go_default_library",
        "//pkg/util/version:go_default_library",
        "//vendor/github.com/armon/circbuf:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
        "//vendor/google.golang.org/grpc:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//vendor/k8s.io/client-go/tools/record:go_default_library",
        "//vendor/k8s.io/client-go/tools/reference:go_default_library",
        "//vendor/k8s.io/client-go/util/flowcontrol:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//pkg/kubelet/qos:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "//pkg/kubelet/apis:go_default_library",
            "//vendor/github.com/docker/docker/pkg/sysinfo:go_default_library",
        ],
        "//conditions:default": [],
    }),
)

go_test(
    name = "go_default_test",
    srcs = [
        "helpers_test.go",
        "instrumented_services_test.go",
        "kuberuntime_container_test.go",
        "kuberuntime_gc_test.go",
        "kuberuntime_image_test.go",
        "kuberuntime_manager_test.go",
        "kuberuntime_sandbox_test.go",
        "labels_test.go",
        "legacy_test.go",
        "security_context_test.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
            "kuberuntime_container_linux_test.go",
        ],
        "//conditions:default": [],
    }),
    embed = [":go_default_library"],
    deps = [
        "//pkg/credentialprovider:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/kubelet/apis/cri/runtime/v1alpha2:go_default_library",
        "//pkg/kubelet/apis/cri/testing:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/container/testing:go_default_library",
        "//pkg/kubelet/lifecycle:go_default_library",
        "//pkg/kubelet/metrics:go_default_library",
        "//vendor/github.com/golang/mock/gomock:go_default_library",
        "//vendor/github.com/google/cadvisor/info/v1:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/github.com/stretchr/testify/require:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/util/feature/testing:go_default_library",
        "//vendor/k8s.io/client-go/util/flowcontrol:go_default_library",
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
        "//pkg/kubelet/kuberuntime/logs:all-srcs",
    ],
    tags = ["automanaged"],
)
