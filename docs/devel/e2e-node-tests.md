<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/docs/devel/e2e-node-tests.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Node End-To-End tests

Node e2e tests are component tests meant for testing the Kubelet code on a custom host environment.

Tests can be run either locally or against a host running on GCE.

Node e2e tests are run as both pre- and post- submit tests by the Kubernetes project.

*Note: Linux only. Mac and Windows unsupported.*

# Running tests

## Locally

Why run tests *Locally*?  Much faster than running tests Remotely.

Prerequisites:
- [Install etcd](https://github.com/coreos/etcd/releases) on your PATH
  - Verify etcd is installed correctly by running `which etcd`
  - Or make etcd binary available and executable at `/tmp/etcd`
- [Install ginkgo](https://github.com/onsi/ginkgo) on your PATH
  - Verify ginkgo is installed correctly by running `which ginkgo`

From the Kubernetes base directory, run:

```sh
make test-e2e-node
```

This will: run the *ginkgo* binary against the subdirectory *test/e2e_node*, which will in turn:
- Ask for sudo access (needed for running some of the processes)
- Build the Kubernetes source code
- Pre-pull docker images used by the tests
- Start a local instance of *etcd*
- Start a local instance of *kube-apiserver*
- Start a local instance of *kubelet*
- Run the test using the locally started processes
- Output the test results to STDOUT
- Stop *kubelet*, *kube-apiserver*, and *etcd*

## Remotely

Why Run tests *Remotely*?  Tests will be run in a customized pristine environment.  Closely mimics what will be done
as pre- and post- submit testing performed by the project.

Prerequisites:
- [join the googlegroup](https://groups.google.com/forum/#!forum/kubernetes-dev)
`kubernetes-dev@googlegroups.com`
  - *This provides read access to the node test images.*
- Setup a [Google Cloud Platform](https://cloud.google.com/) account and project with Google Compute Engine enabled
- Install and setup the [gcloud sdk](https://cloud.google.com/sdk/downloads)
  - Verify the sdk is setup correctly by running `gcloud compute instances list` and `gcloud compute images list --project kubernetes-node-e2e-images`

Run:

```sh
make test-e2e-node REMOTE=true
```

This will:
- Build the Kubernetes source code
- Create a new GCE instance using the default test image
  - Instance will be called **test-e2e-node-containervm-v20160321-image**
- Lookup the instance public ip address
- Copy a compressed archive file to the host containing the following binaries:
  - ginkgo
  - kubelet
  - kube-apiserver
  - e2e_node.test (this binary contains the actual tests to be run)
- Unzip the archive to a directory under **/tmp/gcloud**
- Run the tests using the `ginkgo` command
  - Starts etcd, kube-apiserver, kubelet
  - The ginkgo command is used because this supports more features than running the test binary directly
- Output the remote test results to STDOUT
- `scp` the log files back to the local host under /tmp/_artifacts/e2e-node-containervm-v20160321-image
- Stop the processes on the remote host
- **Leave the GCE instance running**

**Note: Subsequent tests run using the same image will *reuse the existing host* instead of deleting it and
provisioning a new one.  To delete the GCE instance after each test see
*[DELETE_INSTANCE](#delete-instance-after-tests-run)*.**


# Additional Remote Options

## Run tests using different images

This is useful if you want to run tests against a host using a different OS distro or container runtime than
provided by the default image.

List the available test images using gcloud.

```sh
make test-e2e-node LIST_IMAGES=true
```

This will output a list of the available images for the default image project.

Then run:

```sh
make test-e2e-node REMOTE=true IMAGES="<comma-separated-list-images>"
```

## Run tests against a running GCE instance (not an image)

This is useful if you have an host instance running already and want to run the tests there instead of on a new instance.

```sh
make test-e2e-node REMOTE=true HOSTS="<comma-separated-list-of-hostnames>"
```

## Delete instance after tests run

This is useful if you want recreate the instance for each test run to trigger flakes related to starting the instance.

```sh
make test-e2e-node REMOTE=true DELETE_INSTANCES=true
```

## Keep instance, test binaries, and *processes* around after tests run

This is useful if you want to manually inspect or debug the kubelet process run as part of the tests.

```sh
make test-e2e-node REMOTE=true CLEANUP=false
```

## Run tests using an image in another project

This is useful if you want to create your own host image in another project and use it for testing.

```sh
make test-e2e-node REMOTE=true IMAGE_PROJECT="<name-of-project-with-images>" IMAGES="<image-name>"
```

Setting up your own host image may require additional steps such as installing etcd or docker.  See
[setup_host.sh](../../test/e2e_node/environment/setup_host.sh) for common steps to setup hosts to run node tests.

## Create instances using a different instance name prefix

This is useful if you want to create instances using a different name so that you can run multiple copies of the
test in parallel against different instances of the same image.

```sh
make test-e2e-node REMOTE=true INSTANCE_PREFIX="my-prefix"
```

# Additional Test Options for both Remote and Local execution

## Only run a subset of the tests

To run tests matching a regex:

```sh
make test-e2e-node REMOTE=true FOCUS="<regex-to-match>"
```

To run tests NOT matching a regex:

```sh
make test-e2e-node REMOTE=true SKIP="<regex-to-match>"
```

## Run tests continually until they fail

This is useful if you are trying to debug a flaky test failure.  This will cause ginkgo to continually
run the tests until they fail.  **Note: this will only perform test setup once (e.g. creating the instance) and is
less useful for catching flakes related creating the instance from an image.**

```sh
make test-e2e-node REMOTE=true RUN_UNTIL_FAILURE=true
```

## Run tests in parallel

Running test in parallel can usually shorten the test duration. By default node
e2e test runs with`--nodes=8` (see ginkgo flag
[--nodes](https://onsi.github.io/ginkgo/#parallel-specs)). You can use the
`PARALLELISM` option to change the parallelism.

```sh
make test-e2e-node PARALLELISM=4 # run test with 4 parallel nodes
make test-e2e-node PARALLELISM=1 # run test sequentially
```

## Run tests with kubenet network plugin

[kubenet](http://kubernetes.io/docs/admin/network-plugins/#kubenet) is
the default network plugin used by kubelet since Kubernetes 1.3.  The
plugin requires [CNI](https://github.com/containernetworking/cni) and
[nsenter](http://man7.org/linux/man-pages/man1/nsenter.1.html).

Currently, kubenet is enabled by default for Remote execution `REMOTE=true`,
but disabled for Local execution.  **Note: kubenet is not supported for
local execution currently. This may cause network related test result to be
different for Local and Remote execution. So if you want to run network
related test, Remote execution is recommended.**

To enable/disable kubenet:

```sh
make test_e2e_node TEST_ARGS="--disable-kubenet=true" # enable kubenet
make test_e2e_node TEST_ARGS="--disable-kubenet=false" # disable kubenet
```

## Additional QoS Cgroups Hierarchy level testing

For testing with the QoS Cgroup Hierarchy enabled, you can pass --cgroups-per-qos flag as an argument into Ginkgo using TEST_ARGS

```sh
make test_e2e_node TEST_ARGS="--cgroups-per-qos=true"
```

# Notes on tests run by the Kubernetes project during pre-, post- submit.

The node e2e tests are run by the PR builder for each Pull Request and the results published at
the bottom of the comments section.  To re-run just the node e2e tests from the PR builder add the comment
`@k8s-bot node e2e test this issue: #<Flake-Issue-Number or IGNORE>` and **include a link to the test
failure logs if caused by a flake.**

The PR builder runs tests against the images listed in [jenkins-pull.properties](../../test/e2e_node/jenkins/jenkins-pull.properties)

The post submit tests run against the images listed in [jenkins-ci.properties](../../test/e2e_node/jenkins/jenkins-ci.properties)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/e2e-node-tests.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
