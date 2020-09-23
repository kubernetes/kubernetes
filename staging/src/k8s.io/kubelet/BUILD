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
        "//staging/src/k8s.io/kubelet/config/v1beta1:all-srcs",
        "//staging/src/k8s.io/kubelet/pkg/apis/deviceplugin/v1alpha:all-srcs",
        "//staging/src/k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1:all-srcs",
        "//staging/src/k8s.io/kubelet/pkg/apis/pluginregistration/v1:all-srcs",
        "//staging/src/k8s.io/kubelet/pkg/apis/pluginregistration/v1alpha1:all-srcs",
        "//staging/src/k8s.io/kubelet/pkg/apis/pluginregistration/v1beta1:all-srcs",
        "//staging/src/k8s.io/kubelet/pkg/apis/podresources/v1alpha1:all-srcs",
        "//staging/src/k8s.io/kubelet/pkg/apis/stats/v1alpha1:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
