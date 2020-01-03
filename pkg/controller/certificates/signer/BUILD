package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["signer_test.go"],
    data = [
        "testdata/ca.crt",
        "testdata/ca.key",
        "testdata/kubelet.csr",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/api/certificates/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/clock:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//staging/src/k8s.io/client-go/util/cert:go_default_library",
        "//vendor/github.com/google/go-cmp/cmp:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "ca_provider.go",
        "signer.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/controller/certificates/signer",
    deps = [
        "//pkg/apis/certificates/v1beta1:go_default_library",
        "//pkg/controller/certificates:go_default_library",
        "//pkg/controller/certificates/authority:go_default_library",
        "//staging/src/k8s.io/api/certificates/v1beta1:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/dynamiccertificates:go_default_library",
        "//staging/src/k8s.io/client-go/informers/certificates/v1beta1:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/util/cert:go_default_library",
        "//staging/src/k8s.io/client-go/util/keyutil:go_default_library",
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
        "//pkg/controller/certificates/signer/config:all-srcs",
    ],
    tags = ["automanaged"],
)
