# Hunting flaky tests in Kubernetes
Sometimes unit tests are flaky.  This means that due to (usually) race conditions, they will occasionally fail, even though most of the time they pass.

We have a goal of 99.9% flake free tests.  This means that there is only one flake in one thousand runs of a test.

Running a test 1000 times on your own machine can be tedious and time consuming.  Fortunately, there is a better way to achieve this using Kubernetes.

_Note: these instructions are mildly hacky for now, as we get run once semantics and logging they will get better_

There is a testing image ```brendanburns/flake``` up on the docker hub.  We will use this image to test our fix.

Create a replication controller with the following config:
```yaml
id: flakeController
desiredState:
  replicas: 24
  replicaSelector:
    name: flake
  podTemplate:
    desiredState:
      manifest:
        version: v1beta1
        id: ""
        volumes: []
        containers:
        - name: flake
          image: brendanburns/flake
          env:
          - name: TEST_PACKAGE
            value: pkg/tools
          - name: REPO_SPEC
            value: https://github.com/GoogleCloudPlatform/kubernetes
      restartpolicy: {}
    labels:
      name: flake
labels:
  name: flake
```

```./cluster/kubecfg.sh -c controller.yaml create replicaControllers```

This will spin up 100 instances of the test.  They will run to completion, then exit, the kubelet will restart them, eventually you will have sufficient
runs for your purposes, and you can stop the replication controller:

```sh
./cluster/kubecfg.sh stop flakeController
./cluster/kubecfg.sh rm flakeController
```

Now examine the machines with ```docker ps -a``` and look for tasks that exited with non-zero exit codes (ignore those that exited -1, since that's what happens when you stop the replica controller)

Happy flake hunting!
