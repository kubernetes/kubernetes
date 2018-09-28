load("//build:workspace_mirror.bzl", "mirror")
load("//build:workspace.bzl", "CRI_TOOLS_VERSION")

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "97cf62bdef33519412167fd1e4b0810a318a7c234f5f8dc4f53e2da86241c492",
    urls = mirror("https://github.com/bazelbuild/rules_go/releases/download/0.15.3/rules_go-0.15.3.tar.gz"),
)

http_archive(
    name = "io_kubernetes_build",
    sha256 = "1188feb932cefad328b0a3dd75b3ebd1d79dd26dbdd723f019ceb760e27ba6d8",
    strip_prefix = "repo-infra-84d52408a061e87d45aebf5a0867246bdf66d180",
    urls = mirror("https://github.com/kubernetes/repo-infra/archive/84d52408a061e87d45aebf5a0867246bdf66d180.tar.gz"),
)

http_archive(
    name = "bazel_skylib",
    sha256 = "bbccf674aa441c266df9894182d80de104cabd19be98be002f6d478aaa31574d",
    strip_prefix = "bazel-skylib-2169ae1c374aab4a09aa90e65efe1a3aad4e279b",
    urls = mirror("https://github.com/bazelbuild/bazel-skylib/archive/2169ae1c374aab4a09aa90e65efe1a3aad4e279b.tar.gz"),
)

ETCD_VERSION = "3.2.24"

new_http_archive(
    name = "com_coreos_etcd",
    build_file = "third_party/etcd.BUILD",
    sha256 = "947849dbcfa13927c81236fb76a7c01d587bbab42ab1e807184cd91b026ebed7",
    strip_prefix = "etcd-v%s-linux-amd64" % ETCD_VERSION,
    urls = mirror("https://github.com/coreos/etcd/releases/download/v%s/etcd-v%s-linux-amd64.tar.gz" % (ETCD_VERSION, ETCD_VERSION)),
)

http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "29d109605e0d6f9c892584f07275b8c9260803bf0c6fcb7de2623b2bedc910bd",
    strip_prefix = "rules_docker-0.5.1",
    urls = mirror("https://github.com/bazelbuild/rules_docker/archive/v0.5.1.tar.gz"),
)

load("@bazel_skylib//:lib.bzl", "versions")

versions.check(minimum_bazel_version = "0.16.0")

load("@io_bazel_rules_go//go:def.bzl", "go_download_sdk", "go_register_toolchains", "go_rules_dependencies")
load("@io_bazel_rules_docker//docker:docker.bzl", "docker_pull", "docker_repositories")

go_rules_dependencies()

go_register_toolchains(
    go_version = "1.10.4",
)

docker_repositories()

http_file(
    name = "kubernetes_cni",
    sha256 = "f04339a21b8edf76d415e7f17b620e63b8f37a76b2f706671587ab6464411f2d",
    urls = mirror("https://storage.googleapis.com/kubernetes-release/network-plugins/cni-plugins-amd64-v0.6.0.tgz"),
)

http_file(
    name = "cri_tools",
    sha256 = "ccf83574556793ceb01717dc91c66b70f183c60c2bbec70283939aae8fdef768",
    urls = mirror("https://github.com/kubernetes-incubator/cri-tools/releases/download/v%s/crictl-v%s-linux-amd64.tar.gz" % (CRI_TOOLS_VERSION, CRI_TOOLS_VERSION)),
)

docker_pull(
    name = "debian-iptables-amd64",
    digest = "sha256:0987db7ce42949d20ed2647a65d4bee0b616b4d40c7ea54769cc24b7ad003677",
    registry = "k8s.gcr.io",
    repository = "debian-iptables-amd64",
    tag = "v10.2",  # ignored, but kept here for documentation
)

docker_pull(
    name = "debian-hyperkube-base-amd64",
    digest = "sha256:c50522965140c9f206900bf47d547d601c04943e1e59801ba5f70235773cfbb6",
    registry = "k8s.gcr.io",
    repository = "debian-hyperkube-base-amd64",
    tag = "0.10.2",  # ignored, but kept here for documentation
)

docker_pull(
    name = "official_busybox",
    digest = "sha256:cb63aa0641a885f54de20f61d152187419e8f6b159ed11a251a09d115fdff9bd",
    registry = "index.docker.io",
    repository = "library/busybox",
    tag = "latest",  # ignored, but kept here for documentation
)

load("//build:workspace_mirror.bzl", "export_urls")

export_urls("workspace_urls")
