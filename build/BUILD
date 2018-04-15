load("//build:deepcopy.bzl", "k8s_deepcopy_all", "k8s_defaulter_all", "k8s_conversion_all")

package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_docker//docker:docker.bzl", "docker_build", "docker_bundle")
load("@io_kubernetes_build//defs:build.bzl", "release_filegroup")

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//build/debs:all-srcs",
        "//build/release-tars:all-srcs",
        "//build/rpms:all-srcs",
        "//build/visible_to:all-srcs",
    ],
    tags = ["automanaged"],
)

# This list should roughly match kube::build::get_docker_wrapped_binaries()
# in build/common.sh.
DOCKERIZED_BINARIES = {
    "cloud-controller-manager": {
        "base": "@official_busybox//image",
        "target": "//cmd/cloud-controller-manager:cloud-controller-manager",
    },
    "kube-apiserver": {
        "base": "@official_busybox//image",
        "target": "//cmd/kube-apiserver:kube-apiserver",
    },
    "kube-controller-manager": {
        "base": "@official_busybox//image",
        "target": "//cmd/kube-controller-manager:kube-controller-manager",
    },
    "kube-scheduler": {
        "base": "@official_busybox//image",
        "target": "//cmd/kube-scheduler:kube-scheduler",
    },
    "kube-proxy": {
        "base": "@debian-iptables-amd64//image",
        "target": "//cmd/kube-proxy:kube-proxy",
    },
}

[docker_build(
    name = binary + "-internal",
    base = meta["base"],
    cmd = ["/usr/bin/" + binary],
    debs = [
        "//build/debs:%s.deb" % binary,
    ],
    symlinks = {
        # Some cluster startup scripts expect to find the binaries in /usr/local/bin,
        # but the debs install the binaries into /usr/bin.
        "/usr/local/bin/" + binary: "/usr/bin/" + binary,
    },
) for binary, meta in DOCKERIZED_BINARIES.items()]

[docker_bundle(
    name = binary,
    # TODO(thockin): remove the google_containers name after release 1.11.
    images = {
        "k8s.gcr.io/%s:{STABLE_DOCKER_TAG}" % binary: binary + "-internal",
        "gcr.io/google_containers/%s:{STABLE_DOCKER_TAG}" % binary: binary + "-internal",
    },
    stamp = True,
) for binary in DOCKERIZED_BINARIES.keys()]

[genrule(
    name = binary + "_docker_tag",
    srcs = [meta["target"]],
    outs = [binary + ".docker_tag"],
    cmd = "grep ^STABLE_DOCKER_TAG bazel-out/stable-status.txt | awk '{print $$2}' >$@",
    stamp = 1,
) for binary, meta in DOCKERIZED_BINARIES.items()]

genrule(
    name = "os_package_version",
    outs = ["version"],
    cmd = """
grep ^STABLE_BUILD_SCM_REVISION bazel-out/stable-status.txt \
    | awk '{print $$2}' \
    | sed -e 's/^v//' -Ee 's/-([a-z]+)/~\\1/' -e 's/-/+/g' \
    >$@
""",
    stamp = 1,
)

genrule(
    name = "cni_package_version",
    outs = ["cni_version"],
    cmd = "echo 0.5.1 >$@",
)

release_filegroup(
    name = "docker-artifacts",
    srcs = [":%s.tar" % binary for binary in DOCKERIZED_BINARIES.keys()] +
           [":%s.docker_tag" % binary for binary in DOCKERIZED_BINARIES.keys()],
)

# KUBE_CLIENT_TARGETS
release_filegroup(
    name = "client-targets",
    srcs = [
        "//cmd/kubectl",
    ],
)

# KUBE_NODE_TARGETS
release_filegroup(
    name = "node-targets",
    srcs = [
        "//cmd/kube-proxy",
        "//cmd/kubeadm",
        "//cmd/kubelet",
    ],
)

# KUBE_SERVER_TARGETS
# No need to duplicate CLIENT_TARGETS or NODE_TARGETS here,
# since we include them in the actual build rule.
release_filegroup(
    name = "server-targets",
    srcs = [
        "//cluster/gce/gci/mounter",
        "//cmd/cloud-controller-manager",
        "//cmd/hyperkube",
        "//cmd/kube-apiserver",
        "//cmd/kube-controller-manager",
        "//cmd/kube-scheduler",
        "//vendor/k8s.io/kube-aggregator",
    ],
)

# kube::golang::test_targets
filegroup(
    name = "test-targets",
    srcs = [
        "//cmd/gendocs",
        "//cmd/genkubedocs",
        "//cmd/genman",
        "//cmd/genswaggertypedocs",
        "//cmd/genyaml",
        "//cmd/kubemark",  # TODO: server platforms only
        "//cmd/linkcheck",
        "//test/e2e:e2e.test",
        "//test/e2e_node:e2e_node.test",  # TODO: server platforms only
        "//vendor/github.com/onsi/ginkgo/ginkgo",
    ],
)

# KUBE_TEST_PORTABLE
filegroup(
    name = "test-portable-targets",
    srcs = [
        "//hack:e2e.go",
        "//hack:get-build.sh",
        "//hack:ginkgo-e2e.sh",
        "//hack/e2e-internal:all-srcs",
        "//hack/lib:all-srcs",
        "//test/e2e/testing-manifests:all-srcs",
        "//test/kubemark:all-srcs",
    ],
)

filegroup(
    name = "all-generated-sources",
    srcs = [
        ":conversion-sources",
        ":deepcopy-sources",
        ":defaulter-sources",
    ],
)

k8s_deepcopy_all(
    name = "deepcopy-sources",
    packages = [
        "cmd/kubeadm/app/apis/kubeadm",
        "cmd/kubeadm/app/apis/kubeadm/v1alpha1",
        "cmd/kubeadm/app/phases/etcd/spec",
        "pkg/apis/abac",
        "pkg/apis/abac/v0",
        "pkg/apis/abac/v1beta1",
        "pkg/apis/admission",
        "pkg/apis/admissionregistration",
        "pkg/apis/apps",
        "pkg/apis/authentication",
        "pkg/apis/authorization",
        "pkg/apis/autoscaling",
        "pkg/apis/batch",
        "pkg/apis/certificates",
        "pkg/apis/componentconfig",
        "pkg/apis/componentconfig/v1alpha1",
        "pkg/apis/core",
        "pkg/apis/extensions",
        "pkg/apis/imagepolicy",
        "pkg/apis/networking",
        "pkg/apis/policy",
        "pkg/apis/rbac",
        "pkg/apis/scheduling",
        "pkg/apis/settings",
        "pkg/apis/storage",
        "pkg/controller/garbagecollector/metaonly",
        "pkg/kubectl/cmd/testing",
        "pkg/kubelet/apis/kubeletconfig",
        "pkg/kubelet/apis/kubeletconfig/v1beta1",
        "pkg/proxy/apis/kubeproxyconfig",
        "pkg/proxy/apis/kubeproxyconfig/v1alpha1",
        "pkg/registry/rbac/reconciliation",
        "pkg/scheduler/api",
        "pkg/scheduler/api/v1",
        "plugin/pkg/admission/eventratelimit/apis/eventratelimit",
        "plugin/pkg/admission/eventratelimit/apis/eventratelimit/v1alpha1",
        "plugin/pkg/admission/podtolerationrestriction/apis/podtolerationrestriction",
        "plugin/pkg/admission/podtolerationrestriction/apis/podtolerationrestriction/v1alpha1",
        "plugin/pkg/admission/resourcequota/apis/resourcequota",
        "plugin/pkg/admission/resourcequota/apis/resourcequota/v1alpha1",
        "vendor/k8s.io/api/admission/v1beta1",
        "vendor/k8s.io/api/admissionregistration/v1alpha1",
        "vendor/k8s.io/api/admissionregistration/v1beta1",
        "vendor/k8s.io/api/apps/v1",
        "vendor/k8s.io/api/apps/v1beta1",
        "vendor/k8s.io/api/apps/v1beta2",
        "vendor/k8s.io/api/authentication/v1",
        "vendor/k8s.io/api/authentication/v1beta1",
        "vendor/k8s.io/api/authorization/v1",
        "vendor/k8s.io/api/authorization/v1beta1",
        "vendor/k8s.io/api/autoscaling/v1",
        "vendor/k8s.io/api/autoscaling/v2beta1",
        "vendor/k8s.io/api/batch/v1",
        "vendor/k8s.io/api/batch/v1beta1",
        "vendor/k8s.io/api/batch/v2alpha1",
        "vendor/k8s.io/api/certificates/v1beta1",
        "vendor/k8s.io/api/core/v1",
        "vendor/k8s.io/api/events/v1beta1",
        "vendor/k8s.io/api/extensions/v1beta1",
        "vendor/k8s.io/api/imagepolicy/v1alpha1",
        "vendor/k8s.io/api/networking/v1",
        "vendor/k8s.io/api/policy/v1beta1",
        "vendor/k8s.io/api/rbac/v1",
        "vendor/k8s.io/api/rbac/v1alpha1",
        "vendor/k8s.io/api/rbac/v1beta1",
        "vendor/k8s.io/api/scheduling/v1alpha1",
        "vendor/k8s.io/api/settings/v1alpha1",
        "vendor/k8s.io/api/storage/v1",
        "vendor/k8s.io/api/storage/v1alpha1",
        "vendor/k8s.io/api/storage/v1beta1",
        "vendor/k8s.io/apiextensions-apiserver/examples/client-go/pkg/apis/cr/v1",
        "vendor/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions",
        "vendor/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1",
        "vendor/k8s.io/apimachinery/pkg/api/resource",
        "vendor/k8s.io/apimachinery/pkg/apis/meta/internalversion",
        "vendor/k8s.io/apimachinery/pkg/apis/meta/v1",
        "vendor/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured",
        "vendor/k8s.io/apimachinery/pkg/apis/meta/v1beta1",
        "vendor/k8s.io/apimachinery/pkg/apis/testapigroup",
        "vendor/k8s.io/apimachinery/pkg/apis/testapigroup/v1",
        "vendor/k8s.io/apimachinery/pkg/labels",
        "vendor/k8s.io/apimachinery/pkg/runtime",
        "vendor/k8s.io/apimachinery/pkg/runtime/serializer/testing",
        "vendor/k8s.io/apimachinery/pkg/runtime/testing",
        "vendor/k8s.io/apimachinery/pkg/test",
        "vendor/k8s.io/apimachinery/pkg/watch",
        "vendor/k8s.io/apiserver/pkg/admission/plugin/webhook/config/apis/webhookadmission",
        "vendor/k8s.io/apiserver/pkg/admission/plugin/webhook/config/apis/webhookadmission/v1alpha1",
        "vendor/k8s.io/apiserver/pkg/apis/apiserver",
        "vendor/k8s.io/apiserver/pkg/apis/apiserver/v1alpha1",
        "vendor/k8s.io/apiserver/pkg/apis/audit",
        "vendor/k8s.io/apiserver/pkg/apis/audit/v1alpha1",
        "vendor/k8s.io/apiserver/pkg/apis/audit/v1beta1",
        "vendor/k8s.io/apiserver/pkg/apis/example",
        "vendor/k8s.io/apiserver/pkg/apis/example/v1",
        "vendor/k8s.io/apiserver/pkg/apis/example2",
        "vendor/k8s.io/apiserver/pkg/apis/example2/v1",
        "vendor/k8s.io/apiserver/pkg/endpoints/openapi/testing",
        "vendor/k8s.io/apiserver/pkg/endpoints/testing",
        "vendor/k8s.io/apiserver/pkg/registry/rest",
        "vendor/k8s.io/apiserver/pkg/storage/testing",
        "vendor/k8s.io/client-go/pkg/apis/clientauthentication",
        "vendor/k8s.io/client-go/pkg/apis/clientauthentication/v1alpha1",
        "vendor/k8s.io/client-go/rest",
        "vendor/k8s.io/client-go/scale/scheme",
        "vendor/k8s.io/client-go/tools/clientcmd/api",
        "vendor/k8s.io/client-go/tools/clientcmd/api/v1",
        "vendor/k8s.io/code-generator/_examples/apiserver/apis/example",
        "vendor/k8s.io/code-generator/_examples/apiserver/apis/example/v1",
        "vendor/k8s.io/code-generator/_examples/apiserver/apis/example2",
        "vendor/k8s.io/code-generator/_examples/apiserver/apis/example2/v1",
        "vendor/k8s.io/code-generator/_examples/crd/apis/example/v1",
        "vendor/k8s.io/code-generator/_examples/crd/apis/example2/v1",
        "vendor/k8s.io/kube-aggregator/pkg/apis/apiregistration",
        "vendor/k8s.io/kube-aggregator/pkg/apis/apiregistration/v1",
        "vendor/k8s.io/kube-aggregator/pkg/apis/apiregistration/v1beta1",
        "vendor/k8s.io/metrics/pkg/apis/custom_metrics",
        "vendor/k8s.io/metrics/pkg/apis/custom_metrics/v1beta1",
        "vendor/k8s.io/metrics/pkg/apis/external_metrics",
        "vendor/k8s.io/metrics/pkg/apis/external_metrics/v1beta1",
        "vendor/k8s.io/metrics/pkg/apis/metrics",
        "vendor/k8s.io/metrics/pkg/apis/metrics/v1alpha1",
        "vendor/k8s.io/metrics/pkg/apis/metrics/v1beta1",
        "vendor/k8s.io/sample-apiserver/pkg/apis/wardle",
        "vendor/k8s.io/sample-apiserver/pkg/apis/wardle/v1alpha1",
        "vendor/k8s.io/sample-controller/pkg/apis/samplecontroller/v1alpha1",
    ],
)

k8s_defaulter_all(
    name = "defaulter-sources",
    packages = [
        "cmd/kubeadm/app/apis/kubeadm/v1alpha1",
        "pkg/apis/abac/v1beta1",
        "pkg/apis/admission/v1beta1",
        "pkg/apis/admissionregistration/v1alpha1",
        "pkg/apis/admissionregistration/v1beta1",
        "pkg/apis/apps/v1",
        "pkg/apis/apps/v1beta1",
        "pkg/apis/apps/v1beta2",
        "pkg/apis/authentication/v1",
        "pkg/apis/authentication/v1beta1",
        "pkg/apis/authorization/v1",
        "pkg/apis/authorization/v1beta1",
        "pkg/apis/autoscaling/v1",
        "pkg/apis/autoscaling/v2beta1",
        "pkg/apis/batch/v1",
        "pkg/apis/batch/v1beta1",
        "pkg/apis/batch/v2alpha1",
        "pkg/apis/certificates/v1beta1",
        "pkg/apis/componentconfig/v1alpha1",
        "pkg/apis/core/v1",
        "pkg/apis/events/v1beta1",
        "pkg/apis/extensions/v1beta1",
        "pkg/apis/imagepolicy/v1alpha1",
        "pkg/apis/networking/v1",
        "pkg/apis/policy/v1beta1",
        "pkg/apis/rbac/v1",
        "pkg/apis/rbac/v1alpha1",
        "pkg/apis/rbac/v1beta1",
        "pkg/apis/scheduling/v1alpha1",
        "pkg/apis/settings/v1alpha1",
        "pkg/apis/storage/v1",
        "pkg/apis/storage/v1alpha1",
        "pkg/apis/storage/v1beta1",
        "pkg/kubelet/apis/kubeletconfig/v1beta1",
        "pkg/proxy/apis/kubeproxyconfig/v1alpha1",
        "plugin/pkg/admission/eventratelimit/apis/eventratelimit/v1alpha1",
        "plugin/pkg/admission/podtolerationrestriction/apis/podtolerationrestriction/v1alpha1",
        "plugin/pkg/admission/resourcequota/apis/resourcequota/v1alpha1",
        "vendor/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1",
        "vendor/k8s.io/apimachinery/pkg/apis/meta/v1",
        "vendor/k8s.io/apimachinery/pkg/apis/meta/v1beta1",
        "vendor/k8s.io/apimachinery/pkg/apis/testapigroup/v1",
        "vendor/k8s.io/apiserver/pkg/admission/plugin/webhook/config/apis/webhookadmission/v1alpha1",
        "vendor/k8s.io/apiserver/pkg/apis/apiserver/v1alpha1",
        "vendor/k8s.io/apiserver/pkg/apis/audit/v1alpha1",
        "vendor/k8s.io/apiserver/pkg/apis/audit/v1beta1",
        "vendor/k8s.io/apiserver/pkg/apis/example/v1",
        "vendor/k8s.io/apiserver/pkg/apis/example2/v1",
        "vendor/k8s.io/client-go/pkg/apis/clientauthentication/v1alpha1",
    ],
)

k8s_conversion_all(
    name = "conversion-sources",
    packages = [
        "cmd/kubeadm/app/apis/kubeadm/v1alpha1",
        "pkg/apis/abac/v1beta1",
        "pkg/apis/admission/v1beta1",
        "pkg/apis/admissionregistration/v1alpha1",
        "pkg/apis/admissionregistration/v1beta1",
        "pkg/apis/apps/v1",
        "pkg/apis/apps/v1beta1",
        "pkg/apis/apps/v1beta2",
        "pkg/apis/authentication/v1",
        "pkg/apis/authentication/v1beta1",
        "pkg/apis/authorization/v1",
        "pkg/apis/authorization/v1beta1",
        "pkg/apis/autoscaling/v1",
        "pkg/apis/autoscaling/v2beta1",
        "pkg/apis/batch/v1",
        "pkg/apis/batch/v1beta1",
        "pkg/apis/batch/v2alpha1",
        "pkg/apis/certificates/v1beta1",
        "pkg/apis/componentconfig/v1alpha1",
        "pkg/apis/core/v1",
        "pkg/apis/events/v1beta1",
        "pkg/apis/extensions/v1beta1",
        "pkg/apis/imagepolicy/v1alpha1",
        "pkg/apis/networking/v1",
        "pkg/apis/policy/v1beta1",
        "pkg/apis/rbac/v1",
        "pkg/apis/rbac/v1alpha1",
        "pkg/apis/rbac/v1beta1",
        "pkg/apis/scheduling/v1alpha1",
        "pkg/apis/settings/v1alpha1",
        "pkg/apis/storage/v1",
        "pkg/apis/storage/v1alpha1",
        "pkg/apis/storage/v1beta1",
        "pkg/kubelet/apis/kubeletconfig/v1beta1",
        "pkg/proxy/apis/kubeproxyconfig/v1alpha1",
        "plugin/pkg/admission/eventratelimit/apis/eventratelimit/v1alpha1",
        "plugin/pkg/admission/podtolerationrestriction/apis/podtolerationrestriction/v1alpha1",
        "plugin/pkg/admission/resourcequota/apis/resourcequota/v1alpha1",
        "vendor/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1",
        "vendor/k8s.io/apimachinery/pkg/apis/meta/internalversion",
        "vendor/k8s.io/apimachinery/pkg/apis/testapigroup/v1",
        "vendor/k8s.io/apiserver/pkg/admission/plugin/webhook/config/apis/webhookadmission/v1alpha1",
        "vendor/k8s.io/apiserver/pkg/apis/apiserver/v1alpha1",
        "vendor/k8s.io/apiserver/pkg/apis/audit/v1alpha1",
        "vendor/k8s.io/apiserver/pkg/apis/audit/v1beta1",
        "vendor/k8s.io/apiserver/pkg/apis/example/v1",
        "vendor/k8s.io/apiserver/pkg/apis/example2/v1",
        "vendor/k8s.io/client-go/pkg/apis/clientauthentication/v1alpha1",
        "vendor/k8s.io/client-go/scale/scheme/appsv1beta1",
        "vendor/k8s.io/client-go/scale/scheme/appsv1beta2",
        "vendor/k8s.io/client-go/scale/scheme/autoscalingv1",
        "vendor/k8s.io/client-go/scale/scheme/extensionsv1beta1",
        "vendor/k8s.io/code-generator/_examples/apiserver/apis/example/v1",
        "vendor/k8s.io/code-generator/_examples/apiserver/apis/example2/v1",
        "vendor/k8s.io/kube-aggregator/pkg/apis/apiregistration/v1",
        "vendor/k8s.io/kube-aggregator/pkg/apis/apiregistration/v1beta1",
        "vendor/k8s.io/metrics/pkg/apis/custom_metrics/v1beta1",
        "vendor/k8s.io/metrics/pkg/apis/external_metrics/v1beta1",
        "vendor/k8s.io/metrics/pkg/apis/metrics/v1alpha1",
        "vendor/k8s.io/metrics/pkg/apis/metrics/v1beta1",
        "vendor/k8s.io/sample-apiserver/pkg/apis/wardle/v1alpha1",
    ],
)
