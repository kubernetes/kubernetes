package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "arguments.go",
        "endpoint.go",
        "error.go",
        "marshal.go",
        "template.go",
        "version.go",
    ],
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/scheme:go_default_library",
        "//vendor/k8s.io/kubeadm/app/apis/kubeadm:go_default_library",
        "//vendor/k8s.io/kubeadm/app/preflight:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "arguments_test.go",
        "endpoint_test.go",
        "error_test.go",
        "template_test.go",
        "version_test.go",
    ],
    library = ":go_default_library",
    deps = [
        "//vendor/k8s.io/kubeadm/app/apis/kubeadm:go_default_library",
        "//vendor/k8s.io/kubeadm/app/preflight:go_default_library",
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
        "//staging/src/k8s.io/kubeadm/app/util/apiclient:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/util/config:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/util/dryrun:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/util/kubeconfig:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/util/pubkeypin:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/util/staticpod:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/util/token:all-srcs",
    ],
    tags = ["automanaged"],
)
