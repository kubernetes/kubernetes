### Addon-manager

Addon-manager manages two classes of addons with given template files.
- Addons with label `addonmanager.kubernetes.io/reconcile=true` will be periodically
reconciled. Direct manipulation to these addons through Apiserver is discouraged because
Addon-manager will bring them back to the original state, in particular:
	- If an addon is deleted, it will be recreated automatically.
	- If an addon is updated through Apiserver, it will be reconfigured to the state given
	by the supplied fields in the template file.
	- Corresponding addon resource will be deleted when the manifest file is deleted.
- Addons with label `addonmanager.kubernetes.io/ensure-exist=true` will be ensured exist.
User can edit these addons as they want. In particular:
	- They will only be created/re-created with the given template files when there is no
	such resource exist.
	- They will not be deleted when the manifest files are deleted.

Notes:
- Label `kubernetes.io/cluster-service=true` is deprecated (only for Addon Manager).
In future release (after k8s v1.6), Addon Manager may not respect it anymore. Addons
with this label but without `addonmanager.kubernetes.io/ensure-exist=true` will be
treated as "reconcile class addons" for now.
- Resource under $ADDON_PATH (default `/etc/kubernetes/addons/`) needs to have either one
of these two labels (not include the deprecated label) mentioned above. Meanwhile namespaced
resource needs to be in `kube-system` namespace. Otherwise it will be omitted. 
- The above label and namespace rule does not stand for `/opt/namespace.yaml` and
resources under `/etc/kubernetes/admission-controls/`. Addon manager will attempt to create
them regardless during startup.

#### How to release

The `addon-manager` is built for multiple architectures.

1. Change something in the source
2. Bump `VERSION` in the `Makefile`
3. Bump `KUBECTL_VERSION` in the `Makefile` if required
4. Build the `amd64` image and test it on a cluster
5. Push all images

```console
# Build for linux/amd64 (default)
$ make push ARCH=amd64
# ---> gcr.io/google-containers/kube-addon-manager-amd64:VERSION
# ---> gcr.io/google-containers/kube-addon-manager:VERSION (image with backwards-compatible naming)

$ make push ARCH=arm
# ---> gcr.io/google-containers/kube-addon-manager-arm:VERSION

$ make push ARCH=arm64
# ---> gcr.io/google-containers/kube-addon-manager-arm64:VERSION

$ make push ARCH=ppc64le
# ---> gcr.io/google-containers/kube-addon-manager-ppc64le:VERSION

$ make push ARCH=s390x
# ---> gcr.io/google-containers/kube-addon-manager-s390x:VERSION
```

If you don't want to push the images, run `make` or `make build` instead


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/addon-manager/README.md?pixel)]()
