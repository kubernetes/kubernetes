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
[here](http://releases.k8s.io/release-1.2/docs/devel/flaky-tests.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Flaky tests

Any test that fails occasionally is "flaky". Since our merges only proceed when
all tests are green, and we have a number of different CI systems running the
tests in various combinations, even a small percentage of flakes results in a
lot of pain for people waiting for their PRs to merge.

Therefore, it's very important that we write tests defensively. Situations that
"almost never happen" happen with some regularity when run thousands of times in
resource-constrained environments. Since flakes can often be quite hard to
reproduce while still being common enough to block merges occasionally, it's
additionally important that the test logs be useful for narrowing down exactly
what caused the failure.

Note that flakes can occur in unit tests, integration tests, or end-to-end
tests, but probably occur most commonly in end-to-end tests.

## Filing issues for flaky tests

Because flakes may be rare, it's very important that all relevant logs be
discoverable from the issue.

1. Search for the test name. If you find an open issue and you're 90% sure the
   flake is exactly the same, add a comment instead of making a new issue.
2. If you make a new issue, you should title it with the test name, prefixed by
   "e2e/unit/integration flake:" (whichever is appropriate)
3. Reference any old issues you found in step one. Also, make a comment in the
   old issue referencing your new issue, because people monitoring only their
   email do not see the backlinks github adds. Alternatively, tag the person or
   people who most recently worked on it.
4. Paste, in block quotes, the entire log of the individual failing test, not
   just the failure line.
5. Link to durable storage with the rest of the logs. This means (for all the
   tests that Google runs) the GCS link is mandatory! The Jenkins test result
   link is nice but strictly optional: not only does it expire more quickly,
   it's not accesible to non-Googlers.

## Expectations when a flaky test is assigned to you

Note that we won't randomly assign these issues to you unless you've opted in or
you're part of a group that has opted in. We are more than happy to accept help
from anyone in fixing these, but due to the severity of the problem when merges
are blocked, we need reasonably quick turn-around time on test flakes. Therefore
we have the following guidelines:

1. If a flaky test is assigned to you, it's more important than anything else
   you're doing unless you can get a special dispensation (in which case it will
   be reassigned).  If you have too many flaky tests assigned to you, or you
   have such a dispensation, then it's *still* your responsibility to find new
   owners (this may just mean giving stuff back to the relevant Team or SIG Lead).
2. You should make a reasonable effort to reproduce it. Somewhere between an
   hour and half a day of concentrated effort is "reasonable". It is perfectly
   reasonable to ask for help!
3. If you can reproduce it (or it's obvious from the logs what happened), you
   should then be able to fix it, or in the case where someone is clearly more
   qualified to fix it, reassign it with very clear instructions.
4. If you can't reproduce it: __don't just close it!__ Every time a flake comes
   back, at least 2 hours of merge time is wasted. So we need to make monotonic
   progress towards narrowing it down every time a flake occurs. If you can't
   figure it out from the logs, add log messages that would have help you figure
   it out.

# Reproducing unit test flakes

Try the [stress command](https://godoc.org/golang.org/x/tools/cmd/stress).

Just

```
$ go install golang.org/x/tools/cmd/stress
```

Then build your test binary

```
$ go test -c -race
```

Then run it under stress

```
$ stress ./package.test -test.run=FlakyTest
```

It runs the command and writes output to `/tmp/gostress-*` files when it fails.
It periodically reports with run counts. Be careful with tests that use the
`net/http/httptest` package; they could exhaust the available ports on your
system!

# Hunting flaky unit tests in Kubernetes

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
  echo "Checking kubernetes-node-${i}"
  echo "kubernetes-node-${i}:" >> output.txt
  gcloud compute ssh "kubernetes-node-${i}" --command="sudo docker ps -a" >> output.txt
done
grep "Exited ([^0])" output.txt
```

Eventually you will have sufficient runs for your purposes. At that point you can delete the replication controller by running:

```sh
kubectl delete replicationcontroller flakecontroller
```

If you do a final check for flakes with `docker ps -a`, ignore tasks that exited -1, since that's what happens when you stop the replication controller.

Happy flake hunting!


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/flaky-tests.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
