package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "apparmor.go",
        "cassandra.go",
        "configmaps.go",
        "etcd.go",
        "horizontal_pod_autoscalers.go",
        "ingress.go",
        "kube_proxy_migration.go",
        "mysql.go",
        "nvidia-gpu.go",
        "secrets.go",
        "services.go",
        "sysctl.go",
        "upgrade.go",
    ],
    importpath = "k8s.io/kubernetes/test/e2e/upgrades",
    deps = [
        "//pkg/kubelet/sysctl:go_default_library",
        "//staging/src/k8s.io/api/autoscaling/v1:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/extensions/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/version:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//test/e2e/common:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//test/e2e/scheduling:go_default_library",
        "//test/utils/image:go_default_library",
        "//vendor/github.com/davecgh/go-spew/spew:go_default_library",
        "//vendor/github.com/onsi/ginkgo:go_default_library",
        "//vendor/github.com/onsi/gomega:go_default_library",
        "//vendor/github.com/onsi/gomega/gstruct:go_default_library",
        "//vendor/google.golang.org/api/compute/v1:go_default_library",
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
        "//test/e2e/upgrades/apps:all-srcs",
        "//test/e2e/upgrades/storage:all-srcs",
    ],
    tags = ["automanaged"],
)
