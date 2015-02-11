# Hunting flaky tests in Kubernetes
Sometimes unit tests are flaky.  This means that due to (usually) race conditions, they will occasionally fail, even though most of the time they pass.

We have a goal of 99.9% flake free tests.  This means that there is only one flake in one thousand runs of a test.

Running a test 1000 times on your own machine can be tedious and time consuming.  Fortunately, there is a better way to achieve this using Kubernetes.

_Note: these instructions are mildly hacky for now, as we get run once semantics and logging they will get better_

There is a testing image ```brendanburns/flake``` up on the docker hub.  We will use this image to test our fix.

Create a replication controller with the following config:
```yaml
id: flakecontroller
kind: ReplicationController
apiVersion: v1beta1
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

```./cluster/kubectl.sh create -f controller.yaml```

This will spin up 24 instances of the test.  They will run to completion, then exit, and the kubelet will restart them, accumulating more and more runs of the test.
You can examine the recent runs of the test by calling ```docker ps -a``` and looking for tasks that exited with non-zero exit codes. Unfortunately, docker ps -a only keeps around the exit status of the last 15-20 containers with the same image, so you have to check them frequently.
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
./cluster/kubectl.sh stop replicationcontroller flakecontroller
```

If you do a final check for flakes with ```docker ps -a```, ignore tasks that exited -1, since that's what happens when you stop the replication controller.

Happy flake hunting!
