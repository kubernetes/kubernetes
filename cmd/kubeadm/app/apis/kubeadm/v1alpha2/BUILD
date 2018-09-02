load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "bootstraptokenstring.go",
        "conversion.go",
        "defaults.go",
        "defaults_unix.go",
        "defaults_windows.go",
        "doc.go",
        "register.go",
        "types.go",
        "zz_generated.conversion.go",
        "zz_generated.deepcopy.go",
        "zz_generated.defaults.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha2",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/componentconfigs:go_default_library",
        "//cmd/kubeadm/app/constants:go_default_library",
        "//pkg/kubelet/apis/config:go_default_library",
        "//pkg/kubelet/apis/config/scheme:go_default_library",
        "//pkg/kubelet/apis/config/v1beta1:go_default_library",
        "//pkg/proxy/apis/config:go_default_library",
        "//pkg/proxy/apis/config/scheme:go_default_library",
        "//pkg/proxy/apis/config/v1alpha1:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/conversion:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/client-go/tools/bootstrap/token/api:go_default_library",
        "//staging/src/k8s.io/client-go/tools/bootstrap/token/util:go_default_library",
        "//staging/src/k8s.io/kube-proxy/config/v1alpha1:go_default_library",
        "//staging/src/k8s.io/kubelet/config/v1beta1:go_default_library",
        "//vendor/k8s.io/utils/pointer:go_default_library",
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
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["bootstraptokenstring_test.go"],
    embed = [":go_default_library"],
)
