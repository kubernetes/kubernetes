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
        "//plugin/cmd/kube-scheduler:all-srcs",
        "//plugin/pkg/admission/admit:all-srcs",
        "//plugin/pkg/admission/alwayspullimages:all-srcs",
        "//plugin/pkg/admission/antiaffinity:all-srcs",
        "//plugin/pkg/admission/deny:all-srcs",
        "//plugin/pkg/admission/exec:all-srcs",
        "//plugin/pkg/admission/gc:all-srcs",
        "//plugin/pkg/admission/imagepolicy:all-srcs",
        "//plugin/pkg/admission/initialresources:all-srcs",
        "//plugin/pkg/admission/limitranger:all-srcs",
        "//plugin/pkg/admission/namespace/autoprovision:all-srcs",
        "//plugin/pkg/admission/namespace/exists:all-srcs",
        "//plugin/pkg/admission/namespace/lifecycle:all-srcs",
        "//plugin/pkg/admission/persistentvolume/label:all-srcs",
        "//plugin/pkg/admission/podnodeselector:all-srcs",
        "//plugin/pkg/admission/resourcequota:all-srcs",
        "//plugin/pkg/admission/security:all-srcs",
        "//plugin/pkg/admission/securitycontext/scdeny:all-srcs",
        "//plugin/pkg/admission/serviceaccount:all-srcs",
        "//plugin/pkg/admission/storageclass/default:all-srcs",
        "//plugin/pkg/auth:all-srcs",
        "//plugin/pkg/client/auth:all-srcs",
        "//plugin/pkg/scheduler:all-srcs",
    ],
    tags = ["automanaged"],
)
