# This is the list of dependencies allowed to use kubectl.
package_group(
    name = "kubectl_allowed_deps",
    packages = [
        # Kubectl can self-reference itself:
        "//pkg/kubectl/...",
        "//cmd/kubectl/...",
        # It's ok for tests to depend on kubectl.
        "//test/...",
        # Unwelcome dependencies. Let's reduce that list:
        "//cmd/clicheck",
        "//cmd/gendocs",
        "//cmd/genman",
        "//cmd/genyaml",
        "//cmd/gke-certificates-controller",
        "//cmd/hyperkube",
        "//cmd/kube-proxy/app",
        "//cmd/kubeadm/app",
        "//cmd/kubeadm/app/cmd",
        "//federation/cmd/genfeddocs",
        "//federation/cmd/kubefed/app",
        "//federation/pkg/kubefed",
        "//federation/pkg/kubefed/init",
        "//federation/pkg/kubefed/testing",
        "//federation/pkg/kubefed/util",
        "//pkg/printers/internalversion",
    ],
)
