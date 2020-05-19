# gazelle gets confused by our staging/ directory, prepending an extra
# "k8s.io/kubernetes/staging/src" to the importpath.
# Using this directive ensures that it sets the correct importpath.
# gazelle:prefix
# gazelle:importmap_prefix k8s.io/kubernetes/vendor

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
        "//staging/src/k8s.io/api:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver:all-srcs",
        "//staging/src/k8s.io/apimachinery:all-srcs",
        "//staging/src/k8s.io/apiserver:all-srcs",
        "//staging/src/k8s.io/cli-runtime:all-srcs",
        "//staging/src/k8s.io/client-go:all-srcs",
        "//staging/src/k8s.io/cloud-provider:all-srcs",
        "//staging/src/k8s.io/cluster-bootstrap:all-srcs",
        "//staging/src/k8s.io/code-generator:all-srcs",
        "//staging/src/k8s.io/component-base:all-srcs",
        "//staging/src/k8s.io/cri-api:all-srcs",
        "//staging/src/k8s.io/csi-translation-lib:all-srcs",
        "//staging/src/k8s.io/kube-aggregator:all-srcs",
        "//staging/src/k8s.io/kube-controller-manager:all-srcs",
        "//staging/src/k8s.io/kube-proxy:all-srcs",
        "//staging/src/k8s.io/kube-scheduler:all-srcs",
        "//staging/src/k8s.io/kubectl:all-srcs",
        "//staging/src/k8s.io/kubelet:all-srcs",
        "//staging/src/k8s.io/legacy-cloud-providers:all-srcs",
        "//staging/src/k8s.io/metrics:all-srcs",
        "//staging/src/k8s.io/sample-apiserver:all-srcs",
        "//staging/src/k8s.io/sample-cli-plugin:all-srcs",
        "//staging/src/k8s.io/sample-controller:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
