# Update Namespaced Sysctls (GKE Node System Config)

## Background

This script generates namespaced sysctls for a given kernel version. These are
tracked in GoB in
[gke/cluster/gce/gci/sysctl/namespaced-sysctl-names.yaml](https://gke-internal.googlesource.com/kubernetes/+/refs/heads/master/gke/cluster/gce/gci/sysctl/namespaced-sysctl-names.yaml)
and used as part of setting sysctls druing node startup.

GKE maintains
[gke/cluster/gce/gci/sysctl/sysctl-defaults.yaml](https://gke-internal.git.corp.google.com/kubernetes/+/refs/heads/master/gke/cluster/gce/gci/sysctl/sysctl-defaults.yaml)
which contains the sysctls which should be set by default during node startup.
Sysctls can be namespaced or not namespaced. If the sysctls are namespaced, it
needs to be set twice, once on the root namespace, and second inside the
namespace for each container. This former is done as part of `configure.sh`
startup scripts, the latter by kubelet.

To determine if a sysctl is namespaced, the sysctl is looked up in
[gke/cluster/gce/gci/sysctl/namespaced-sysctl-names.yaml](https://gke-internal.googlesource.com/kubernetes/+/refs/heads/master/gke/cluster/gce/gci/sysctl/namespaced-sysctl-names.yaml)
for a given kernel version. Sysctls may become namespaced in newer kernel
versions, and as such, periodically for new kernel versions, we should
regenerate the list of namespaced sysctls. Generating the list of namespaced
sysctls is the goal of this script.

Some relevant docs for more background:

* [go/gke-node-config](http://goto.google.com/gke-node-config)
* [go/gke-node-kernel-params-tuning](http://gke-node-kernel-params-tuning)
* [go/gke-node-config-notes](http://goto.google.com/gke-node-config-notes)

## Process

1. Ensure you have gcloud setup and a test GCP project you can use to spin
   up a test VM
2. Pick the image to test. This will likely be from
   [`image_maps.textpb`](http://google3/production/borg/cloud-kubernetes/config/image_maps.textpb)
   and find version / image to use. The image project to use will likely be

* `gke-node-images`
* `ubuntu-os-gke-cloud`

Since sysctls are dependent on the underlying kernel, using an image directly
from `cos-cloud` project or `ubuntu-cloud` should be fine as well, as long as
the kernel version matches the GKE image being used.

4. Run the script to generate the namespaced sysctls:

```
# ensure PWD is KUBEROOT
$ ./gke/cluster/gce/gci/sysctl-internal/update-namespaced-sysctls/run.sh --image "${IMAGE}" --image_project "${IMAGE_PROJECT}" --project "${PROJECT}" --zone "${ZONE}"
```

This will take a bit, since it will spin up a GCE VM using the provided image,
run the `generate_namespaced.py` on the VM, and scp the results back to your
machine. When it's done, it will dump the namespaced sysctls file to a
directory in `/tmp` on your local machine, and print the path of the generated file.

5. Run the merge script to merge the newly generated namespaced sysctls with
   the existing `namespaced-sysctl-names.yaml` stored in GoB.

```
$ python3 gke/cluster/gce/gci/sysctl-internal/update-namespaced-sysctls/merge.py --new-namespaced-sysctls /tmp/namespaced_sysctls_sysctl-test-vm-{some-guid}.yaml --existing-namespaced-sysctls gke/cluster/gce/gci/sysctl/namespaced-sysctl-names.yaml
```

6. The script will display a diff of the sysctls that have changed in the new
   kernel version. Check if the diff looks reasonable, and if so send a CL with
   the newly updated `namespaced-sysctl-names.yaml`.

## Example

For example, let's assume as checked into GoB currently,
[gke/cluster/gce/gci/sysctl/namespaced-sysctl-names.yaml](https://gke-internal.googlesource.com/kubernetes/+/refs/heads/master/gke/cluster/gce/gci/sysctl/namespaced-sysctl-names.yaml)
newest version were generated on 4.19 kernel, but new shiny COS release with
5.4 kernel was released and we want to generate the namespaced sysctls for 5.4
kernel.

We'll test on `cos-85-lts` which is on 5.4 kernel. COS describes LTS version to
kernel mapping on the [release
notes](https://cloud.google.com/container-optimized-os/docs/release-notes).

Let's find the image to use.

```
$ gcloud compute images list --project="cos-cloud" --no-standard-images
NAME                         PROJECT    FAMILY      DEPRECATED  STATUS
cos-77-12371-1105-0          cos-cloud  cos-77-lts              READY
cos-81-12871-1226-0          cos-cloud  cos-81-lts              READY
cos-85-13310-1041-38         cos-cloud  cos-85-lts              READY
cos-beta-85-13310-1041-1     cos-cloud  cos-beta                READY
cos-dev-88-15823-0-0         cos-cloud  cos-dev                 READY
cos-stable-85-13310-1041-38  cos-cloud  cos-stable              READY
```

We will use `cos-85-lts` family, which is `cos-85-13310-1041-38`.

Running the `run.sh` script as follows:

```
$ IMAGE="cos-85-13310-1041-38"
$ IMAGE_PROJECT="cos-cloud"
$ PROJECT="porterdavid-gke-dev" # any project you can spin up a VM in will work.
$ ZONE="us-central1-c"
$ ./gke/cluster/gce/gci/sysctl-internal/update-namespaced-sysctls/run.sh --image "${IMAGE}" --image_project "${IMAGE_PROJECT}" --project "${PROJECT}" --zone "${ZONE}"
Starting sysctl-test-vm-38af8f4d-5195-4130-9e3f-99912dfeed42...
[cut]
All done! Output is at /tmp/namespaced_sysctls_sysctl-test-vm-38af8f4d-5195-4130-9e3f-99912dfeed42.yaml
```

Now we merge the newly generated sysctls with the existing file stored in GoB.

```
$ python3 gke/cluster/gce/gci/sysctl-internal/update-namespaced-sysctls/merge.py --new-namespaced-sysctls /tmp/namespaced_sysctls_sysctl-test-vm-38af8f4d-5195-4130-9e3f-99912dfeed42.yaml --existing-namespaced-sysctls gke/cluster/gce/gci/sysctl/namespaced-sysctl-names.yaml
```

It will look something like this:

```
$ python3 gke/cluster/gce/gci/sysctl-internal/update-namespaced-sysctls/merge.py --new-namespaced-sysctls /tmp/namespaced_sysctls_sysctl-test-vm-38af8f4d-5195-4130-9e3f-99912dfeed42.yaml --existing-namespaced-sysctls gke/cluster/gce/gci/sysctl/namespaced-sysctl-names.yaml
INFO:root:New namespaced sysctls are on kernel version 5.4
INFO:root:Diffing new sysctls with kernel version: 5.4 with existing sysctls on kernel version: 4.19
Added:
        net.ipv6.icmp.echo_ignore_multicast
        net.netfilter.nf_conntrack_gre_timeout_stream
        net.ipv6.route.skip_notify_on_dev_down
        net.ipv4.raw_l3mdev_accept
        net.netfilter.nf_conntrack_gre_timeout
        net.ipv6.icmp.echo_ignore_anycast
Removed:
        net.core.android_paranoid
        net.ipv6.conf.all.accept_ra_rt_table
        net.ipv6.conf.default.accept_ra_rt_table
        net.ipv4.tcp_default_init_rwnd
Does the diff look reasonable? Continue with the merge?
Please remember to include the diff below in a CL updating the namespaced sysctls (y/N)
```

The resulting diff is printed. Sysctls are added when the upstream kernel introduces new sysctls.
Sysctls can be removed for two reasons:

1. The sysctl is removed from the new kernel.
2. The sysctl becomes unnamespaced in the new kernel.

With respect to sysctl removals, sysctls becoming unnamespaced in a new kernel
(option 2) will likely never happen. Most likely if a sysctl is removed, this
was due to the sysctl being removed entirely from the new kernel.
The main thing to pay attention and be careful of is sysctls removals especially if they're included in
[gke/cluster/gce/gci/sysctl/sysctl-defaults.yaml](https://gke-internal.git.corp.google.com/kubernetes/+/refs/heads/master/gke/cluster/gce/gci/sysctl/sysctl-defaults.yaml).

In this case, the diff looks reasonable, there appears to be a some new
namespaced sysctls and some removed, but none of removed sysctls are in
[gke/cluster/gce/gci/sysctl/sysctl-defaults.yaml](https://gke-internal.git.corp.google.com/kubernetes/+/refs/heads/master/gke/cluster/gce/gci/sysctl/sysctl-defaults.yaml),
so this update will be a no-op and not bring any actual changes in node startup.

We proceed to update the file.

```
Please remember to include the diff below in a CL updating the namespaced sysctls (y/N) y
INFO:root:Done! Updated gke/cluster/gce/gci/sysctl/namespaced-sysctl-names.yaml with the new merged sysctls
```

`gke/cluster/gce/gci/sysctl/namespaced-sysctl-names.yaml` has been updated!
Commit the changes and send a CL to GoB with them.
