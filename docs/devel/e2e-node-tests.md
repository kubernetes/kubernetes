<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.2/docs/devel/e2e-node-tests.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Node End-To-End tests

Node e2e tests start kubelet and minimal supporting infrastructure to validate
the kubelet on a host. Tests can be run either locally, against a remote host or
against a GCE image.

*Note: Linux only. Mac and Windows unsupported.*

## Running tests locally

etcd must be installed and on the PATH to run the node e2e tests.  To verify
etcd is installed: `which etcd`. You can find instructions for installing etcd
[on the etcd releases page](https://github.com/coreos/etcd/releases).

Run the tests locally: `make test_e2e_node`

Running the node e2e tests locally will build the kubernetes go source files and
then start the kubelet, kube-apiserver, and etcd binaries on localhost before
executing the ginkgo tests under test/e2e_node against the local kubelet
instance.

## Running tests against a remote host

The node e2e tests can be run against one or more remote hosts using one of:
* [e2e-node-jenkins.sh](../../test/e2e_node/jenkins/e2e-node-jenkins.sh) (gce
only)
* [run_e2e.go](../../test/e2e_node/runner/run_e2e.go) (requires passwordless ssh
and remote passwordless sudo access over ssh)
* using [run_e2e.go](../../test/e2e_node/runner/run_e2e.go) to build a tar.gz
and executing on host (requires host access w/ remote sudo)

### Configuring a new remote host for testing

The host must contain a environment capable of supporting a mini-kubernetes
cluster. Includes:
* install etcd
* install docker
* install lxc and update grub commandline
* enable tty-less sudo access

See [setup_host.sh](../../test/e2e_node/environment/setup_host.sh)

### Running the tests

1. If running against a host on gce

  * Copy [template.properties](../../test/e2e_node/jenkins/template.properties)

    * Fill in `GCE_HOSTS`
    * Set `INSTALL_GODEP=true` to install `godep`, `gomega`, `ginkgo`

  * Make sure host names are resolvable to ssh `ssh <host>`.

    * If needed, you can run `gcloud compute config-ssh` to add gce hostnames to
your .ssh/config so they are resolvable by ssh.

  * Run `test/e2e_node/jenkins/e2e-node-jenkins.sh <path to properties file>`
    * **Must be run from kubernetes root**

2. If running against a host anywhere else

  * **Requires password-less ssh and sudo access**

    * Make sure this works - e.g. `ssh <hostname> -- sudo echo "ok"`
    * If ssh flags are required (e.g. `-i`), they can be used and passed to the
tests with `--ssh-options`

  * `go run test/e2e_node/runner/run_e2e.go --logtostderr --hosts <comma
separated hosts>`

    * **Must be run from kubernetes root**
    * requires (go get): `github.com/tools/godep`, `github.com/onsi/gomega`,
`github.com/onsi/ginkgo/ginkgo`

3. Alternatively, manually build and copy `e2e_node_test.tar.gz` to a remote
host

  * Build the tar.gz `go run test/e2e_node/runner/run_e2e.go --logtostderr
--build-only`

    * requires (go get): `github.com/tools/godep`, `github.com/onsi/gomega`,
`github.com/onsi/ginkgo/ginkgo`

  * Copy `e2e_node_test.tar.gz` to the remote host

  * Extract the archive on the remote host `tar -xzvf e2e_node_test.tar.gz`

  * Run the tests `./e2e_node.test --logtostderr --vmodule=*=2
--build-services=false --node-name=<hostname>`

      * Note: This must be run from the directory containing the kubelet and
kube-apiserver binaries.

## Running tests against a gce image

* Build a gce image from a prepared gce host
  * Create the host from a base image and configure it (see above)
    * Run tests against this remote host to ensure that it is setup correctly
before doing anything else
  * Create a gce *snapshot* of the instance
  * Create a gce *disk* from the snapshot
  * Create a gce *image* from the disk
* Test that the necessary gcloud credentials are setup for the project
  * `gcloud compute --project <project> --zone <zone> images list`
  * Verify that your image appears in the list
* Copy [template.properties](../../test/e2e_node/jenkins/template.properties)
  * Fill in `GCE_PROJECT`, `GCE_ZONE`, `GCE_IMAGES`
* Run `test/e2e_node/jenkins/e2e-node-jenkins.sh <path to properties file>`
  * **Must be run from kubernetes root**

## Kubernetes Jenkins CI and PR builder

Node e2e tests are run against a static list of host environments continuously
or when manually triggered on a github.com pull requests using the trigger
phrase `@k8s-bot test node e2e experimental` - *results not yet publish, pending
evaluation of test stability.*.


### CI Host environments

TBD

### PR builder host environments

| linux distro    | distro version | docker version | etcd version | cloud provider |
|-----------------|----------------|----------------|--------------|----------------|
| containervm     |                | 1.8            |              | gce            |
| rhel            | 7              | 1.10           |              | gce            |
| centos          | 7              | 1.10           |              | gce            |
| coreos          | stable         | 1.8            |              | gce            |
| debian          | jessie         | 1.10           |              | gce            |
| ubuntu          | trusty         | 1.8            |              | gce            |
| ubuntu          | trusty         | 1.9            |              | gce            |
| ubuntu          | trusty         | 1.10           |              | gce            |
| ubuntu          | wily           | 1.10           |              | gce            |







<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/e2e-node-tests.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
