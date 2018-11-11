load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "cert_pool.go",
        "curves.go",
        "error.go",
        "errors.go",
        "names.go",
        "nilref_nil_darwin.go",
        "nilref_zero_darwin.go",
        "pem_decrypt.go",
        "pkcs1.go",
        "pkcs8.go",
        "ptr_sysptr_windows.go",
        "ptr_uint_windows.go",
        "revoked.go",
        "root.go",
        "root_bsd.go",
        "root_cgo_darwin.go",
        "root_darwin.go",
        "root_darwin_armx.go",
        "root_linux.go",
        "root_nacl.go",
        "root_nocgo_darwin.go",
        "root_plan9.go",
        "root_solaris.go",
        "root_unix.go",
        "root_windows.go",
        "rpki.go",
        "sec1.go",
        "verify.go",
        "x509.go",
    ],
    cgo = True,
    clinkopts = select({
        "@io_bazel_rules_go//go/platform:darwin_386": [
            "-framework CoreFoundation -framework Security",
        ],
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "-framework CoreFoundation -framework Security",
        ],
        "//conditions:default": [],
    }),
    copts = select({
        "@io_bazel_rules_go//go/platform:darwin_386": [
            "-mmacosx-version-min=10.6 -D__MAC_OS_X_VERSION_MAX_ALLOWED=1080",
        ],
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "-mmacosx-version-min=10.6 -D__MAC_OS_X_VERSION_MAX_ALLOWED=1080",
        ],
        "//conditions:default": [],
    }),
    importmap = "k8s.io/kubernetes/vendor/github.com/google/certificate-transparency-go/x509",
    importpath = "github.com/google/certificate-transparency-go/x509",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/google/certificate-transparency-go/asn1:go_default_library",
        "//vendor/github.com/google/certificate-transparency-go/tls:go_default_library",
        "//vendor/github.com/google/certificate-transparency-go/x509/pkix:go_default_library",
        "//vendor/golang.org/x/crypto/cryptobyte:go_default_library",
        "//vendor/golang.org/x/crypto/cryptobyte/asn1:go_default_library",
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
        "//vendor/github.com/google/certificate-transparency-go/x509/pkix:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
