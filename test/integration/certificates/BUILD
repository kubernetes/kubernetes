load("@io_bazel_rules_go//go:def.bzl", "go_test")

go_test(
    name = "go_default_test",
    srcs = [
        "admission_approval_test.go",
        "admission_sign_test.go",
        "admission_subjectrestriction_test.go",
        "admission_test.go",
        "controller_approval_test.go",
        "defaulting_test.go",
        "field_selector_test.go",
        "main_test.go",
    ],
    tags = ["integration"],
    deps = [
        "//cmd/kube-apiserver/app/testing:go_default_library",
        "//pkg/controller/certificates:go_default_library",
        "//pkg/controller/certificates/approver:go_default_library",
        "//staging/src/k8s.io/api/authorization/v1:go_default_library",
        "//staging/src/k8s.io/api/certificates/v1beta1:go_default_library",
        "//staging/src/k8s.io/api/rbac/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/typed/authorization/v1:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/typed/certificates/v1beta1:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//test/integration/framework:go_default_library",
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
