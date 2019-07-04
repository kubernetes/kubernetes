{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/JH35X82)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Queries for Getting or Describing Resources
{% endpanel %}

# Matching Objects from Get and Describing

## Motivation

Match Resources with Queries when Getting or Describing them.

{% method %}
## Resource Config By `kustomization.yaml`

Get all Resources provided by the `kustomization.yaml` in project/.
{% sample lang="yaml" %}

```bash
kubectl get -k project/
```

{% endmethod %}

{% method %}
## Resource Config By Dir

Get all Resources present in the Resource Config for a directory.
{% sample lang="yaml" %}

```bash
kubectl get -f configs/
```

{% endmethod %}

{% method %}
## Resource Types

Get **all** Resources in a namespace for a given type.

The Group and Version for the Resource are determined by the apiserver discovery service.

The Singular, Plural, Short Name also apply to *Types with Name* and *Types with Selectors*.
{% sample lang="yaml" %}

```bash
# Plural
kubectl get deployments
```

```bash
# Singular
kubectl get deployment
```

```bash
# Short name
kubectl get deploy
```

{% endmethod %}

{% method %}
## Resource Types with Group / Version

Get **all** Resources in a namespace for a given type.

The Group and Version for the Resource are explicit.

{% sample lang="yaml" %}

```bash
kubectl get deployments.apps
```

```bash
kubectl get deployments.v1.apps
```

{% endmethod %}

{% method %}
## Resource Types with Name

Get named Resources in a namespace for a given type.

{% sample lang="yaml" %}

```bash
kubectl get deployment nginx
```

{% endmethod %}

{% method %}
## Label Selector

Get **all** Resources in a namespace **matching a label select** for a given type.
{% sample lang="yaml" %}

```bash
kubectl get deployments -l app=nginx
```

{% endmethod %}

{% method %}
## Namespaces

By default Get and Describe will fetch resource in the default namespace or the namespace specified
with `--namespace`.

The `---all-namespaces` flag will **fetch Resources from all namespaces**.

{% sample lang="yaml" %}

```bash
kubectl get deployments --all-namespaces
```

{% endmethod %}


{% method %}
## List multiple Resource types

Get and Describe can accept **multiple Resource types**, and it will print them both in separate sections.

{% sample lang="yaml" %}

```bash
kubectl get deployments,services
```

{% endmethod %}

  
{% method %}
## List multiple Resource types by name

Get and Describe can accept **multiple Resource types and names**.

{% sample lang="yaml" %}

```bash
kubectl get kubectl get rc/web service/frontend pods/web-pod-13je7
```

{% endmethod %}
  
{% method %}
## Uninitialized

Kubernetes **Resources may be hidden until they have gone through an initialization process**.
These Resources can be view with the `--include-uninitialized` flag.

{% sample lang="yaml" %}

```bash
kubectl get deployments --include-uninitialized
```

{% endmethod %}

{% method %}
## Not Found

By default, Get or Describe **will return an error if an object is requested and doesn't exist**.
The `--ignore-not-found` flag will cause kubectl to exit 0 if the Resource is not found

{% sample lang="yaml" %}

```bash
kubectl get deployment nginx --ignore-not-found
```

{% endmethod %}
