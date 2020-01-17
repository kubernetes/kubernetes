# Starting a Windows Kubernetes cluster on GCE using kube-up

## IMPORTANT PLEASE NOTE!
Any time the file structure in the `windows` directory changes, `windows/BUILD`
and `k8s.io/release/lib/releaselib.sh` must be manually updated with the
changes. We HIGHLY recommend not changing the file structure, because consumers
of Kubernetes releases depend on the release structure remaining stable.

## Bring up the cluster

Prerequisites: a Google Cloud Platform project.

### 0. Prepare your environment

Clone this repository under your `$GOPATH/src` directory on a Linux machine.
Then, optionally clean/prepare your environment using these commands:

```
# Remove files that interfere with get-kube / kube-up:
rm -rf ./kubernetes/; rm -f kubernetes.tar.gz; rm -f ~/.kube/config

# Set the default gcloud project for this shell. This is optional but convenient
# if you're working with multiple projects and don't want to repeatedly switch
# between gcloud config configurations.
export CLOUDSDK_CORE_PROJECT=<your_project_name>

# To run e2e test locally, make sure "Application Default Credentials" is set in any of the places:
# References: https://cloud.google.com/sdk/docs/authorizing#authorizing_with_a_service_account
#             https://cloud.google.com/sdk/gcloud/reference/auth/application-default/
#    1. $HOME/.config/gcloud/application_default_credentials.json, if doesn't exist, run this command:
gcloud auth application-default login
# Or 2. Create a json format credential file as per http://cloud/docs/authentication/production,
#       then export to environment variable
export GOOGLE_APPLICATION_CREDENTIAL=[path_to_the_json_file]
```

### 1. Build Kubernetes

NOTE: this step is only needed if you want to test local changes you made to
the codebase.

The most straightforward approach to build those binaries is to run `make
release`. However, that builds binaries for all supported platforms, and can be
slow. You can speed up the process by following the instructions below to only
build the necessary binaries.

```
# Apply https://github.com/yujuhong/kubernetes/commit/27e608a050a997be5ab736a7cdeb29aa68f3b7ee to your tree:
curl \
  https://github.com/yujuhong/kubernetes/commit/27e608a050a997be5ab736a7cdeb29aa68f3b7ee.patch | \
  git apply

# Build binaries for both Linux and Windows:
make quick-release
```

### 2. Create a Kubernetes cluster

You can create a regular Kubernetes cluster or an end-to-end test cluster.<br />
Only end-to-end test clusters support running the Kubernetes e2e tests (as both [e2e cluster creation](https://github.com/kubernetes/kubernetes/blob/b632eaddbaad9dc1430d214d506b72750bbb9f69/hack/e2e-internal/e2e-up.sh#L24) and [e2e test scripts](https://github.com/kubernetes/kubernetes/blob/b632eaddbaad9dc1430d214d506b72750bbb9f69/hack/ginkgo-e2e.sh#L42) are setup based on `cluster/gce/config-test.sh`), also enables some debugging features such as SSH access on the Windows nodes. 

Please make sure you set the environment variables properly following the
instructions in the previous section.

First, set the following environment variables which are required for
controlling the number of Linux and Windows nodes in the cluster and for
enabling IP aliases (which are required for Windows pod routing). At least one
Linux worker node is required and two are recommended because many default
cluster-addons (e.g., `kube-dns`) need to run on Linux nodes. The master control
plane only runs on Linux.

```
export NUM_NODES=2  # number of Linux nodes
export NUM_WINDOWS_NODES=2
export KUBE_GCE_ENABLE_IP_ALIASES=true
export KUBERNETES_NODE_PLATFORM=windows
export LOGGING_STACKDRIVER_RESOURCE_TYPES=new
```

Now bring up a cluster using one of the following two methods:

#### 2a. Create a regular Kubernetes cluster

```
# Invoke kube-up.sh with these environment variables:
#   PROJECT: text name of your GCP project.
#   KUBERNETES_SKIP_CONFIRM: skips any kube-up prompts.
PROJECT=${CLOUDSDK_CORE_PROJECT} KUBERNETES_SKIP_CONFIRM=y ./cluster/kube-up.sh
```

To teardown the cluster run:

```
PROJECT=${CLOUDSDK_CORE_PROJECT} KUBERNETES_SKIP_CONFIRM=y ./cluster/kube-down.sh
```

#### 2b. Create a Kubernetes end-to-end (E2E) test cluster	
If you have built your own release binaries following step 1, run the following	
command:	
```	
PROJECT=${CLOUDSDK_CORE_PROJECT} ./hack/e2e-internal/e2e-up.sh	
```	

If any e2e cluster exists already, this command will prompt you whether tears down and creates a new	one. To teardown existing e2e cluster only, run the command:
```	
PROJECT=${CLOUDSDK_CORE_PROJECT} ./hack/e2e-internal/e2e-down.sh	
```	

No matter what type of cluster you chose to create, the result should be a	
Kubernetes cluster with one Linux master node, `NUM_NODES` Linux worker nodes	
and `NUM_WINDOWS_NODES` Windows worker nodes.	

## Validating the cluster

Invoke this script to run a smoke test that verifies that the cluster has been
brought up correctly:

```
cluster/gce/windows/smoke-test.sh
```

Sometimes the first run of the smoke test will fail because it took too long to
pull the Windows test containers. The smoke test will usually pass on the next
attempt.

## Running e2e tests against the cluster

If you brought up an end-to-end test cluster using the steps above then you can
use the steps below to run K8s e2e tests. These steps are based on
[kubernetes-sigs/windows-testing](https://github.com/kubernetes-sigs/windows-testing).

*   Build the necessary test binaries. This must be done after every change to
    test code.

    ```
    make WHAT=test/e2e/e2e.test
    ```

*   Set necessary environment variables and fetch the `run-e2e.sh` script:

    ```
    export KUBECONFIG=~/.kube/config
    export WORKSPACE=$(pwd)
    export ARTIFACTS=${WORKSPACE}/e2e-artifacts

    curl \
      https://raw.githubusercontent.com/yujuhong/gce-k8s-windows-testing/master/run-e2e.sh \
      -o ${WORKSPACE}/run-e2e.sh
    chmod u+x run-e2e.sh
    ```

    NOTE: `run-e2e.sh` begins with a 5 minute sleep to wait for container images
    to be pre-pulled. You'll probably want to edit the script and remove this.

*   The canonical arguments for running all Windows e2e tests against a cluster
    on GCE can be seen by searching for `--test-cmd-args` in the [test
    configuration](https://github.com/kubernetes/test-infra/blob/master/config/jobs/kubernetes/sig-windows/windows-gce.yaml#L78)
    for the `ci-kubernetes-e2e-windows-gce` continuous test job. These arguments
    should be passed to the `run-e2e` script; escape the ginkgo arguments by
    adding quotes around them. For example:

    ```
    ./run-e2e.sh --node-os-distro=windows \
      --ginkgo.focus="\[Conformance\]|\[NodeConformance\]|\[sig-windows\]" \
      --ginkgo.skip="\[LinuxOnly\]|\[Serial\]|\[Feature:.+\]" --minStartupPods=8
    ```

*   Run a single test by setting the ginkgo focus to match your test name; for
    example, the "DNS should provide DNS for the cluster" test can be run using:

    ```
    ./run-e2e.sh --node-os-distro=windows \
      --ginkgo.focus="provide\sDNS\sfor\sthe\scluster"
    ```

    Make sure to always include `--node-os-distro=windows` for testing against
    Windows nodes.

After the test run completes, log files can be found under the `${ARTIFACTS}`
directory.
