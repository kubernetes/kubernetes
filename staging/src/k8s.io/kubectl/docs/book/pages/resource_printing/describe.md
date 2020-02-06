{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/JH35X82)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Print verbose debug information about a Resource
{% endpanel %}

# Describe Resources

## Motivation

{% method %}
Describe is a **higher level printing operation that may aggregate data from other sources** in addition
to the Resource being queried (e.g. Events).

Describe pulls out the most important information about a Resource from the Resource itself and related
Resources, and formats and prints this information on multiple lines.

- Aggregates data from related Resources
- Formats Verbose Output for debugging

{% sample lang="yaml" %}

```bash
kubectl describe deployments
```

```bash
Name:                   nginx
Namespace:              default
CreationTimestamp:      Thu, 15 Nov 2018 10:58:03 -0800
Labels:                 app=nginx
Annotations:            deployment.kubernetes.io/revision=1
Selector:               app=nginx
Replicas:               1 desired | 1 updated | 1 total | 1 available | 0 unavailable
StrategyType:           RollingUpdate
MinReadySeconds:        0
RollingUpdateStrategy:  25% max unavailable, 25% max surge
Pod Template:
  Labels:  app=nginx
  Containers:
   nginx:
    Image:        nginx
    Port:         <none>
    Host Port:    <none>
    Environment:  <none>
    Mounts:       <none>
  Volumes:        <none>
Conditions:
  Type           Status  Reason
  ----           ------  ------
  Progressing    True    NewReplicaSetAvailable
  Available      True    MinimumReplicasAvailable
OldReplicaSets:  <none>
NewReplicaSet:   nginx-78f5d695bd (1/1 replicas created)
Events:          <none>
```

{% endmethod %}

{% panel style="info", title="Get vs Describe" %}
When Describing a Resource, it may aggregate information from several other Resources.  For instance Describing
a Node will aggregate Pod Resources to print the Node utilization.

When Getting a Resource, it will only print information available from reading that Resource.  While Get may aggregate
data from the *fields* of that Resource, it won't look at fields from other Resources.
{% endpanel %}
