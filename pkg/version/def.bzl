# Implements hack/lib/version.sh's kube::version::ldflags() for Bazel.
def version_x_defs():
  # This should match the list of packages in kube::version::ldflag
  stamp_pkgs = [
      "k8s.io/kubernetes/pkg/version",
      # In hack/lib/version.sh, this has a vendor/ prefix. That isn't needed here?
      "k8s.io/client-go/pkg/version",
      ]
  # This should match the list of vars in kube::version::ldflags
  # It should also match the list of vars set in hack/print-workspace-status.sh.
  stamp_vars = [
      "buildDate",
      "gitCommit",
      "gitMajor",
      "gitMinor",
      "gitTreeState",
      "gitVersion",
  ]
  # Generate the cross-product.
  x_defs = {}
  for pkg in stamp_pkgs:
    for var in stamp_vars:
      x_defs["%s.%s" % (pkg, var)] = "{%s}" % var
  return x_defs
