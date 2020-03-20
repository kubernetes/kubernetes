workspace(name = "io_k8s_kubernetes")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("//build:workspace_mirror.bzl", "mirror")

http_archive(
    name = "bazel_toolchains",
    sha256 = "3a6ffe6dd91ee975f5d5bc5c50b34f58e3881dfac59a7b7aba3323bd8f8571a8",
    strip_prefix = "bazel-toolchains-92dd8a7",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-toolchains/archive/92dd8a7.tar.gz",
        "https://github.com/bazelbuild/bazel-toolchains/archive/92dd8a7.tar.gz",
    ],
)

load("@bazel_toolchains//rules:rbe_repo.bzl", "rbe_autoconfig")

rbe_autoconfig(
    name = "rbe_default",
    base_container_digest = "sha256:677c1317f14c6fd5eba2fd8ec645bfdc5119f64b3e5e944e13c89e0525cc8ad1",
    digest = "sha256:b7c2e7a18968b9df2db43eda722c5ae592aafbf774ba2766074a9c96926743d8",
    registry = "gcr.io",
    repository = "k8s-testimages/bazel-krte",
    # tag = "latest",
)

http_archive(
    name = "bazel_skylib",
    sha256 = "eb5c57e4c12e68c0c20bc774bfbc60a568e800d025557bc4ea022c6479acc867",
    strip_prefix = "bazel-skylib-0.6.0",
    urls = mirror("https://github.com/bazelbuild/bazel-skylib/archive/0.6.0.tar.gz"),
)

load("@bazel_skylib//lib:versions.bzl", "versions")

versions.check(minimum_bazel_version = "0.23.0")

http_archive(
    name = "io_k8s_repo_infra",
    sha256 = "f6d65480241ec0fd7a0d01f432938b97d7395aeb8eefbe859bb877c9b4eafa56",
    strip_prefix = "repo-infra-9f4571ad7242bf3ec4b47365062498c2528f9a5f",
    urls = mirror("https://github.com/kubernetes/repo-infra/archive/9f4571ad7242bf3ec4b47365062498c2528f9a5f.tar.gz"),
)

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "62bedd372f125fe62c16c0cc2ad9d7a2b6a1171d639933a5651a729fdce497fc",
    urls = mirror("https://github.com/bazelbuild/rules_go/releases/download/v0.20.7/rules_go-v0.20.7.tar.gz"),
)

load("@io_bazel_rules_go//go:deps.bzl", "go_download_sdk", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()

# The version of rules_go we're using here is no longer supported, so
# we must manually download a newer version of the go sdk.
go_download_sdk(
    name = "go_sdk",
    sdks = {
        "darwin_amd64": ("go1.13.9.darwin-amd64.tar.gz", "450e59538ed5d3f2b165ba5107530afce6e8e89c6cc5c90a0cbf0a58846ef3b1"),
        "freebsd_386": ("go1.13.9.freebsd-386.tar.gz", "6b75a5a46ebbdf06aa5023f2bd0ad7e9e37389125468243368d5795e1c15c9cd"),
        "freebsd_amd64": ("go1.13.9.freebsd-amd64.tar.gz", "87716246da52c193226df44031aaf45e45ebfc23e01bdc845311c1b560e76e2b"),
        "linux_386": ("go1.13.9.linux-386.tar.gz", "a2744aa2ddc68d888e9f65c2cbe4c8b527b139688ce232ead90dc2961f8d51a8"),
        "linux_amd64": ("go1.13.9.linux-amd64.tar.gz", "f4ad8180dd0aaf7d7cda7e2b0a2bf27e84131320896d376549a7d849ecf237d7"),
        "linux_arm64": ("go1.13.9.linux-arm64.tar.gz", "b53cb466d7986e5e17a3d4c196bc95df08a35968eced5efd7e128387a246c46e"),
        "linux_arm": ("go1.13.9.linux-armv6l.tar.gz", "a3c2941a1fde8692514ece7e2180a0e3ca70609f52756a66bc0ab68c63572361"),
        "linux_ppc64le": ("go1.13.9.linux-ppc64le.tar.gz", "90beb01962202f332be0a7c8dad2db3d30242759ba863db3f36c45d241940efc"),
        "linux_s390x": ("go1.13.9.linux-s390x.tar.gz", "a40949aaf55912b06df8fda511c33fde3e52d377706bdc095332652c1ad225e3"),
        "windows_386": ("go1.13.9.windows-386.zip", "e22406377448f1aea2dd1517327e5ae452d826c0c7624b3511d5af510c57b69a"),
        "windows_amd64": ("go1.13.9.windows-amd64.zip", "cf066aabdf4d83c251aaace14b57a35aafffd1fa67d54d907f27fb31e470a135"),
    },
)

go_register_toolchains()

http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "aed1c249d4ec8f703edddf35cbe9dfaca0b5f5ea6e4cd9e83e99f3b0d1136c3d",
    strip_prefix = "rules_docker-0.7.0",
    urls = mirror("https://github.com/bazelbuild/rules_docker/archive/v0.7.0.tar.gz"),
)

load(
    "@io_bazel_rules_docker//repositories:repositories.bzl",
    container_repositories = "repositories",
)

container_repositories()

load("@io_bazel_rules_docker//container:container.bzl", "container_pull")

container_pull(
    name = "official_busybox",
    digest = "sha256:5e8e0509e829bb8f990249135a36e81a3ecbe94294e7a185cc14616e5fad96bd",
    registry = "index.docker.io",
    repository = "library/busybox",
    tag = "latest",  # ignored, but kept here for documentation
)

load("//build:workspace.bzl", "release_dependencies")

release_dependencies()

load("//build:workspace_mirror.bzl", "export_urls")

export_urls("workspace_urls")
