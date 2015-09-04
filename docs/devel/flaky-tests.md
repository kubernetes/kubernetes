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

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/devel/flaky-tests.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Hunting flaky tests in Kubernetes

Sometimes unit tests are flaky.  This means that due to (usually) race conditions, they will occasionally fail, even though most of the time they pass.

We have a goal of 99.9% flake free tests.  This means that there is only one flake in one thousand runs of a test.

Running a test 1000 times on your own machine can be tedious and time consuming.  Fortunately, there is a better way to achieve this using Kubernetes.

_Note: these instructions are mildly hacky for now, as we get run once semantics and logging they will get better_

There is a testing image `brendanburns/flake` up on the docker hub.  We will use this image to test our fix.

Create a replication controller with the following config:

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: flakecontroller
spec:
  replicas: 24
  template:
    metadata:
      labels:
        name: flake
    spec:
      containers:
      - name: flake
        image: brendanburns/flake
        env:
        - name: TEST_PACKAGE
          value: pkg/tools
        - name: REPO_SPEC
          value: https://github.com/kubernetes/kubernetes
```

Note that we omit the labels and the selector fields of the replication controller, because they will be populated from the labels field of the pod template by default.

```sh
kubectl create -f ./controller.yaml
```

This will spin up 24 instances of the test.  They will run to completion, then exit, and the kubelet will restart them, accumulating more and more runs of the test.
You can examine the recent runs of the test by calling `docker ps -a` and looking for tasks that exited with non-zero exit codes. Unfortunately, docker ps -a only keeps around the exit status of the last 15-20 containers with the same image, so you have to check them frequently.
You can use this script to automate checking for failures, assuming your cluster is running on GCE and has four nodes:

```sh
echo "" > output.txt
for i in {1..4}; do
  echo "Checking kubernetes-minion-${i}"
  echo "kubernetes-minion-${i}:" >> output.txt
  gcloud compute ssh "kubernetes-minion-${i}" --command="sudo docker ps -a" >> output.txt
done
grep "Exited ([^0])" output.txt
```

Eventually you will have sufficient runs for your purposes. At that point you can stop and delete the replication controller by running:

```sh
kubectl stop replicationcontroller flakecontroller
```

If you do a final check for flakes with `docker ps -a`, ignore tasks that exited -1, since that's what happens when you stop the replication controller.

Happy flake hunting!


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/flaky-tests.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
