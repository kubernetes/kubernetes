### Addon-manager

addon-manager manages two classes of addons with given template files in
`$ADDON_PATH` (default `/etc/kubernetes/addons/`).
- Addons with label `addonmanager.kubernetes.io/mode=Reconcile` will be periodically
reconciled. Direct manipulation to these addons through apiserver is discouraged because
addon-manager will bring them back to the original state. In particular:
	- Addon will be re-created if it is deleted.
	- Addon will be reconfigured to the state given by the supplied fields in the template
	file periodically.
	- Addon will be deleted when its manifest file is deleted from the `$ADDON_PATH`.
- Addons with label `addonmanager.kubernetes.io/mode=EnsureExists` will be checked for
existence only. Users can edit these addons as they want. In particular:
	- Addon will only be created/re-created with the given template file when there is no
	instance of the resource with that name.
	- Addon will not be deleted when the manifest file is deleted from the `$ADDON_PATH`.

Notes:
- Label `kubernetes.io/cluster-service=true` is deprecated (only for Addon Manager).
In future release (after one year), Addon Manager may not respect it anymore. Addons
have this label but without `addonmanager.kubernetes.io/mode=EnsureExists` will be
treated as "reconcile class addons" for now.
- Resources under `$ADDON_PATH` need to have either one of these two labels.
Meanwhile namespaced resources need to be in `kube-system` namespace.
Otherwise it will be omitted.
- The above label and namespace rule does not stand for `/opt/namespace.yaml` and
resources under `/etc/kubernetes/admission-controls/`. addon-manager will attempt to
create them regardless during startup.

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
