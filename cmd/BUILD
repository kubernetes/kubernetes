package(default_visibility = ["//visibility:public"])

licenses(["notice"])

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
        "//cmd/clicheck:all-srcs",
        "//cmd/cloud-controller-manager:all-srcs",
        "//cmd/gendocs:all-srcs",
        "//cmd/genkubedocs:all-srcs",
        "//cmd/genman:all-srcs",
        "//cmd/genslateyaml:all-srcs",
        "//cmd/genswaggertypedocs:all-srcs",
        "//cmd/genutils:all-srcs",
        "//cmd/genyaml:all-srcs",
        "//cmd/gke-certificates-controller:all-srcs",
        "//cmd/hyperkube:all-srcs",
        "//cmd/kube-apiserver:all-srcs",
        "//cmd/kube-controller-manager:all-srcs",
        "//cmd/kube-proxy:all-srcs",
        "//cmd/kubeadm:all-srcs",
        "//cmd/kubectl:all-srcs",
        "//cmd/kubelet:all-srcs",
        "//cmd/kubemark:all-srcs",
        "//cmd/libs/go2idl/client-gen:all-srcs",
        "//cmd/libs/go2idl/conversion-gen:all-srcs",
        "//cmd/libs/go2idl/deepcopy-gen:all-srcs",
        "//cmd/libs/go2idl/defaulter-gen:all-srcs",
        "//cmd/libs/go2idl/go-to-protobuf:all-srcs",
        "//cmd/libs/go2idl/import-boss:all-srcs",
        "//cmd/libs/go2idl/informer-gen:all-srcs",
        "//cmd/libs/go2idl/lister-gen:all-srcs",
        "//cmd/libs/go2idl/openapi-gen:all-srcs",
        "//cmd/libs/go2idl/set-gen:all-srcs",
        "//cmd/linkcheck:all-srcs",
        "//cmd/mungedocs:all-srcs",
    ],
    tags = ["automanaged"],
)
