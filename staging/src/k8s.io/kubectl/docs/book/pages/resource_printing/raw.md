{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/JH35X82)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Get or List Raw Resources in a cluster as Yaml or Json
{% endpanel %}

# Print Raw Resource

## Motivation

Inspecting or Debugging Resources.

The Kubernetes Resources stored in etcd by the apiserver have **many more fields than
are shown in the summarized views**.  Users can learn much more about a Resource by
viewing the Raw Resource as Yaml or Json.  The Raw Resource will contain:

- fields specified by the **user** in the Resource Config (e.g. `metadata.name`)
- metadata fields owned by the **apiserver** (e.g. `metadata.creationTimestamp`)
- fields defaulted by the **apiserver** (e.g. `spec..imagePullPolicy`)
- fields set by **Controllers** (e.g. `spec.clusterIp`, `status`)

## Get

The `kubectl get` reads Resources from the cluster and formats them as output.  The examples in
this chapter will query for Resources by providing Get the *Resource Type* as an argument.
For more query options see [Queries and Options](queries_and_options.md).

{% method %}

### YAML

Print the Raw Resource formatting it as YAML.

{% sample lang="yaml" %}

```bash
kubectl get deployments -o yaml
```

```yaml
apiVersion: v1
items:
- apiVersion: extensions/v1beta1
  kind: Deployment
  metadata:
    annotations:
      deployment.kubernetes.io/revision: "1"
    creationTimestamp: 2018-11-15T18:58:03Z
    generation: 1
    labels:
      app: nginx
    name: nginx
    namespace: default
    resourceVersion: "1672574"
    selfLink: /apis/extensions/v1beta1/namespaces/default/deployments/nginx
    uid: 6131547f-e908-11e8-9ff6-42010a8a00d1
  spec:
    progressDeadlineSeconds: 600
    replicas: 1
    revisionHistoryLimit: 10
    selector:
      matchLabels:
        app: nginx
    strategy:
      rollingUpdate:
        maxSurge: 25%
        maxUnavailable: 25%
      type: RollingUpdate
    template:
      metadata:
        creationTimestamp: null
        labels:
          app: nginx
      spec:
        containers:
        - image: nginx
          imagePullPolicy: Always
          name: nginx
          resources: {}
          terminationMessagePath: /dev/termination-log
          terminationMessagePolicy: File
        dnsPolicy: ClusterFirst
        restartPolicy: Always
        schedulerName: default-scheduler
        securityContext: {}
        terminationGracePeriodSeconds: 30
  status:
    availableReplicas: 1
    conditions:
    - lastTransitionTime: 2018-11-15T18:58:10Z
      lastUpdateTime: 2018-11-15T18:58:10Z
      message: Deployment has minimum availability.
      reason: MinimumReplicasAvailable
      status: "True"
      type: Available
    - lastTransitionTime: 2018-11-15T18:58:03Z
      lastUpdateTime: 2018-11-15T18:58:10Z
      message: ReplicaSet "nginx-78f5d695bd" has successfully progressed.
      reason: NewReplicaSetAvailable
      status: "True"
      type: Progressing
    observedGeneration: 1
    readyReplicas: 1
    replicas: 1
    updatedReplicas: 1
kind: List
metadata:
  resourceVersion: ""
  selfLink: ""
```

{% endmethod %}

---

{% method %}

### JSON

Print the Raw Resource formatting it as JSON.

{% sample lang="yaml" %}

```bash
kubectl get deployments -o json
```

```json
{
    "apiVersion": "v1",
    "items": [
        {
            "apiVersion": "extensions/v1beta1",
            "kind": "Deployment",
            "metadata": {
                "annotations": {
                    "deployment.kubernetes.io/revision": "1"
                },
                "creationTimestamp": "2018-11-15T18:58:03Z",
                "generation": 1,
                "labels": {
                    "app": "nginx"
                },
                "name": "nginx",
                "namespace": "default",
                "resourceVersion": "1672574",
                "selfLink": "/apis/extensions/v1beta1/namespaces/default/deployments/nginx",
                "uid": "6131547f-e908-11e8-9ff6-42010a8a00d1"
            },
            "spec": {
                "progressDeadlineSeconds": 600,
                "replicas": 1,
                "revisionHistoryLimit": 10,
                "selector": {
                    "matchLabels": {
                        "app": "nginx"
                    }
                },
                "strategy": {
                    "rollingUpdate": {
                        "maxSurge": "25%",
                        "maxUnavailable": "25%"
                    },
                    "type": "RollingUpdate"
                },
                "template": {
                    "metadata": {
                        "creationTimestamp": null,
                        "labels": {
                            "app": "nginx"
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "image": "nginx",
                                "imagePullPolicy": "Always",
                                "name": "nginx",
                                "resources": {},
                                "terminationMessagePath": "/dev/termination-log",
                                "terminationMessagePolicy": "File"
                            }
                        ],
                        "dnsPolicy": "ClusterFirst",
                        "restartPolicy": "Always",
                        "schedulerName": "default-scheduler",
                        "securityContext": {},
                        "terminationGracePeriodSeconds": 30
                    }
                }
            },
            "status": {
                "availableReplicas": 1,
                "conditions": [
                    {
                        "lastTransitionTime": "2018-11-15T18:58:10Z",
                        "lastUpdateTime": "2018-11-15T18:58:10Z",
                        "message": "Deployment has minimum availability.",
                        "reason": "MinimumReplicasAvailable",
                        "status": "True",
                        "type": "Available"
                    },
                    {
                        "lastTransitionTime": "2018-11-15T18:58:03Z",
                        "lastUpdateTime": "2018-11-15T18:58:10Z",
                        "message": "ReplicaSet \"nginx-78f5d695bd\" has successfully progressed.",
                        "reason": "NewReplicaSetAvailable",
                        "status": "True",
                        "type": "Progressing"
                    }
                ],
                "observedGeneration": 1,
                "readyReplicas": 1,
                "replicas": 1,
                "updatedReplicas": 1
            }
        }
    ],
    "kind": "List",
    "metadata": {
        "resourceVersion": "",
        "selfLink": ""
    }
}
```

{% endmethod %}
