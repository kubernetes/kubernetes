# RuntimeClass

RuntimeClass is an alpha feature for supporting multiple container runtimes within a cluster. When
enabled, pods can select a RuntimeClass to run with using the `PodSpec.RuntimeClassName` field.

To enable RuntimeClass, set the feature gate `RuntimeClass=true`, and ensure the CRD defined in this
directory is installed.

For more information, see:
https://git.k8s.io/enhancements/keps/sig-node/runtime-class.md

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/runtimeclass/README.md?pixel)]()
