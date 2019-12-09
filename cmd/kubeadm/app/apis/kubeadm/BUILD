package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "bootstraptokenhelpers.go",
        "bootstraptokenstring.go",
        "doc.go",
        "register.go",
        "types.go",
        "zz_generated.deepcopy.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm",
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/cluster-bootstrap/token/api:go_default_library",
        "//staging/src/k8s.io/cluster-bootstrap/token/util:go_default_library",
        "//staging/src/k8s.io/cluster-bootstrap/util/secrets:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
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
        "//cmd/kubeadm/app/apis/kubeadm/fuzzer:all-srcs",
        "//cmd/kubeadm/app/apis/kubeadm/scheme:all-srcs",
        "//cmd/kubeadm/app/apis/kubeadm/v1beta1:all-srcs",
        "//cmd/kubeadm/app/apis/kubeadm/v1beta2:all-srcs",
        "//cmd/kubeadm/app/apis/kubeadm/validation:all-srcs",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "bootstraptokenhelpers_test.go",
        "bootstraptokenstring_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
    ],
)
