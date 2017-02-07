package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_test",
)
load(
    "@io_kubernetes_build//defs:go.bzl",
    "api_group",
)

api_group(
    name = "go_default_library",
    srcs = [
        "conversion.go",
        "defaults.go",
        "doc.go",
        "generated.pb.go",
        "register.go",
        "types.generated.go",
        "types.go",
        "types_swagger_doc_generated.go",
    ],
    gen_deps = [
        "//cmd/kubeadm/app/apis/kubeadm/v1alpha1:go_default_library",
        "//federation/apis/federation/v1beta1:go_default_library",
        "//pkg/apis/abac/v1beta1:go_default_library",
        "//pkg/apis/apps/v1beta1:go_default_library",
        "//pkg/apis/authentication/v1beta1:go_default_library",
        "//pkg/apis/authorization/v1:go_default_library",
        "//pkg/apis/authorization/v1beta1:go_default_library",
        "//pkg/apis/autoscaling:go_default_library",
        "//pkg/apis/autoscaling/v1:go_default_library",
        "//pkg/apis/batch:go_default_library",
        "//pkg/apis/batch/v1:go_default_library",
        "//pkg/apis/batch/v2alpha1:go_default_library",
        "//pkg/apis/certificates/v1beta1:go_default_library",
        "//pkg/apis/componentconfig/v1alpha1:go_default_library",
        "//pkg/apis/imagepolicy/v1alpha1:go_default_library",
        "//pkg/apis/rbac/v1alpha1:go_default_library",
        "//pkg/apis/rbac/v1beta1:go_default_library",
        "//pkg/apis/storage/v1beta1:go_default_library",
        "//vendor:k8s.io/apiserver/pkg/apis/example/v1",
    ],
    gen_types = [
        "conversion",
        "deepcopy",
        "defaulter",
    ],
    extra_peers = [
        "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1",
        "k8s.io/kubernetes/federation/apis/federation/v1beta1",
        "k8s.io/kubernetes/pkg/api/v1",
        "k8s.io/kubernetes/pkg/apis/abac/v1beta1",
        "k8s.io/kubernetes/pkg/apis/apps/v1beta1",
        "k8s.io/kubernetes/pkg/apis/authentication/v1beta1",
        "k8s.io/kubernetes/pkg/apis/authorization/v1",
        "k8s.io/kubernetes/pkg/apis/authorization/v1beta1",
        "k8s.io/kubernetes/pkg/apis/autoscaling/v1",
        "k8s.io/kubernetes/pkg/apis/batch/v1",
        "k8s.io/kubernetes/pkg/apis/batch/v2alpha1",
        "k8s.io/kubernetes/pkg/apis/certificates/v1beta1",
        "k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1",
        "k8s.io/kubernetes/pkg/apis/extensions/v1beta1",
        "k8s.io/kubernetes/pkg/apis/imagepolicy/v1alpha1",
        "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1",
        "k8s.io/kubernetes/pkg/apis/rbac/v1beta1",
        "k8s.io/kubernetes/pkg/apis/storage/v1beta1",
        "k8s.io/kubernetes/vendor/k8s.io/apimachinery/pkg/apis/meta/v1",
        "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/apis/example/v1",
    ],
    package = "pkg/apis/extensions/v1beta1",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//vendor:github.com/gogo/protobuf/proto",
        "//vendor:github.com/gogo/protobuf/sortkeys",
        "//vendor:github.com/ugorji/go/codec",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/conversion",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/intstr",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = ["defaults_test.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/install:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/extensions/install:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/intstr",
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
)
