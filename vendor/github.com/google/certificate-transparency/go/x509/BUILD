load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "cert_pool.go",
        "pem_decrypt.go",
        "pkcs1.go",
        "pkcs8.go",
        "root.go",
        "sec1.go",
        "verify.go",
        "x509.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "root_darwin.go",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "root_unix.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "root_windows.go",
        ],
        "//conditions:default": [],
    }),
    cgo = True,
    clinkopts = select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "-framework CoreFoundation -framework Security",
        ],
        "//conditions:default": [],
    }),
    copts = select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "-mmacosx-version-min=10.6 -D__MAC_OS_X_VERSION_MAX_ALLOWED=1060",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/google/certificate-transparency/go/asn1:go_default_library",
        "//vendor/github.com/google/certificate-transparency/go/x509/pkix:go_default_library",
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
        "//vendor/github.com/google/certificate-transparency/go/x509/pkix:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
