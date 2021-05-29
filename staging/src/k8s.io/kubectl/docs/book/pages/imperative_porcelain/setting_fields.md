{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/JH35X82)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Imperatively Set fields on Resources
{% endpanel %}

# Creating Resources

## Motivation

Set fields on Resources directly from the command line for the purposes of development or debugging.
Not for production Application Management.

{% method %}
## Scale

The Replicas field on a Resource can be set using the `kubectl scale` command.


{% sample lang="yaml" %}

```bash
# Scale a replicaset named 'foo' to 3.
kubectl scale --replicas=3 rs/foo
```

```sh
# Scale a resource identified by type and name specified in "foo.yaml" to 3.
kubectl scale --replicas=3 -f foo.yaml
```

```sh
# If the deployment named mysql's current size is 2, scale mysql to 3.
kubectl scale --current-replicas=2 --replicas=3 deployment/mysql
```

```sh
# Scale multiple replication controllers.
kubectl scale --replicas=5 rc/foo rc/bar rc/baz
```

```sh
# Scale statefulset named 'web' to 3.
kubectl scale --replicas=3 statefulset/web
```

{% endmethod %}

{% panel style="info", title="Conditional Scale Update" %}
It is possible to conditionally update the replicas if and only if the
replicas haven't changed from their last known value using the `--current-replicas` flag.
e.g. `kubectl scale --current-replicas=2 --replicas=3 deployment/mysql`
{% endpanel %}


{% method %}
## Labels

Labels can be set using the `kubectl label` command.  Multiple Resources can
be updated in a single command using the `-l` flag.

{% sample lang="yaml" %}

```sh
# Update pod 'foo' with the label 'unhealthy' and the value 'true'.
kubectl label pods foo unhealthy=true
```

```sh
# Update pod 'foo' with the label 'status' and the value 'unhealthy', overwriting any existing value.
kubectl label --overwrite pods foo status=unhealthy
```

```sh
# Update all pods in the namespace
kubectl label pods --all status=unhealthy
```

```sh
# Update a pod identified by the type and name in "pod.json"
kubectl label -f pod.json status=unhealthy
```

```sh
# Update pod 'foo' only if the resource is unchanged from version 1.
kubectl label pods foo status=unhealthy --resource-version=1
```

```sh
# Update pod 'foo' by removing a label named 'bar' if it exists.
# Does not require the --overwrite flag.
kubectl label pods foo bar-
```

{% endmethod %}

{% method %}
## Annotations

Annotations can be set using the `kubectl annotate` command.

{% sample lang="yaml" %}

```sh
# Update pod 'foo' with the annotation 'description' and the value 'my frontend'.
# If the same annotation is set multiple times, only the last value will be applied
kubectl annotate pods foo description='my frontend'
```

```sh
# Update a pod identified by type and name in "pod.json"
kubectl annotate -f pod.json description='my frontend'
```

```sh
# Update pod 'foo' with the annotation 'description' and the value 'my frontend running nginx', overwriting any
existing value.
kubectl annotate --overwrite pods foo description='my frontend running nginx'
```

```sh
# Update all pods in the namespace
kubectl annotate pods --all description='my frontend running nginx'
```

```sh
# Update pod 'foo' only if the resource is unchanged from version 1.
kubectl annotate pods foo description='my frontend running nginx' --resource-version=1
```

```sh
# Update pod 'foo' by removing an annotation named 'description' if it exists.
# Does not require the --overwrite flag.
kubectl annotate pods foo description-
```

{% endmethod %}

{% method %}
## Patches

Arbitrary fields can be set using the `kubectl patch` command.

{% sample lang="yaml" %}

```sh
# Partially update a node using a strategic merge patch. Specify the patch as JSON.
kubectl patch node k8s-node-1 -p '{"spec":{"unschedulable":true}}'
```

```sh
# Partially update a node using a strategic merge patch. Specify the patch as YAML.
kubectl patch node k8s-node-1 -p $'spec:\n unschedulable: true'
```

```sh
# Partially update a node identified by the type and name specified in "node.json" using strategic merge patch.
kubectl patch -f node.json -p '{"spec":{"unschedulable":true}}'
```

```sh
# Update a container's image; spec.containers[*].name is required because it's a merge key.
kubectl patch pod valid-pod -p '{"spec":{"containers":[{"name":"kubernetes-serve-hostname","image":"new image"}]}}'
```

```sh
# Update a container's image using a json patch with positional arrays.
kubectl patch pod valid-pod --type='json' -p='[{"op": "replace", "path": "/spec/containers/0/image", "value":"newimage"}]'
```
{% endmethod %}
