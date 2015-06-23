<!--
Copyright 2015 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

-->
# Limit range example
This example demonstrates the usage of limit ranges in Kubernetes.

### Step Zero: Prerequisites

This example assumes that you have forked the repository and [turned up a Kubernetes cluster](../../docs/getting-started-guides):

```bash
$ cd kubernetes
$ ./cluster/kube-up.sh
```

### Step One: Create a pod

Before you start mucking with things, check out the default limits:

```bash
$ kubectl describe limits limits
Name:           limits
Type            Resource        Min     Max     Default
----            --------        ---     ---     ---
Container       cpu             -       -       100m
```

Then, go ahead and create some pods:

```bash
$ kubectl create -f examples/limitrange/valid-pod.json
pods/valid-pod
$ kubectl create -f examples/limitrange/invalid-pod.json
pods/invalid-pod
```

The `invalid-pod` should succeed for now, but won't for long.

### Step Two: Specify limits

```bash
$ kubectl update -f examples/limitrange/limit-range.json
limitranges/limits
```

### Step Three: Try to update and/or create pods outside of those limits

Try to update it:

```bash
$ kubectl update -f examples/limitrange/invalid-pod.json
Error from server: Pod "invalid-pod" is forbidden: Minimum CPU usage per pod is 250m, but requested 10m
```

Or, delete it and try again:

```bash
$ kubectl delete -f examples/limitrange/invalid-pod.json
pods/invalid-pod
$ kubectl create -f examples/limitrange/invalid-pod.json
Error from server: Pod "invalid-pod" is forbidden: Minimum CPU usage per pod is 250m, but requested 10m
```

A valid pod still works:

```bash
$ kubectl update -f examples/limitrange/valid-pod.json
pods/valid-pod
$ kubectl delete -f examples/limitrange/valid-pod.json
pods/valid-pod
$ kubectl create -f examples/limitrange/valid-pod.json
pods/valid-pod
```

### Step Four: Cleanup

To turn down a Kubernetes cluster:

```bash
$ ./cluster/kube-down.sh
```

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/limitrange/README.md?pixel)]()
