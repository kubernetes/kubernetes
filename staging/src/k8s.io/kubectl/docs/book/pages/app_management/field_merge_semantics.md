{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/CLQBQHR)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Fields set and deleted from Resource Config are merged into Resources by Apply
    - If a Resource already exists, Apply updates the Resources by merging the local Resource Config into the remote Resources
    - Fields removed from the Resource Config will be deleted from the remote Resource
{% endpanel %}

# Merging Fields

{% panel style="warning", title="Advanced Section" %}
This chapter contains advanced material that readers may want to skip and come back to later.
{% endpanel %}

## When are fields merged?

This page describes how Resource Config is merged with Resources or other Resource Config.  This
may occur when:

- Applying Resource Config updates to the live Resources in the cluster 
- Defining Patches in the `kustomization.yaml` which are overlayed on `resources` and [bases](../app_customization/bases_and_variants.md)

### Applying Resource Config Updates

Rather than replacing the Resource with the new Resource Config, **Apply will merge the new Resource Config
into the live Resource**.  This retains values which may be set by the control plane - such as `replicas` values
set by auto scalers

### Defining Patches

`patches` are sparse Resource Config which **contain a subset of fields that override values
defined in other Resource Config** with the same Group/Version/Kind/Namespace/Name.
This is used to alter values defined on Resource Config without having to fork it.

## Motivation (Apply)

This page describes the semantics for merging Resource Config.

Ownership of Resource fields are shared between declarative Resource Config authored by human
users, and values set by Controllers running in the cluster.  Some fields, such as the `status`
and `clusterIp` fields, are owned exclusively by Controllers.  Fields, such as the `name`
and `namespace` fields, are owned exclusively by the human user managing the Resource.

Other fields, such as `replicas`, may be owned by either human users, the apiserver or
Controllers.  For example, `replicas` may be explicitly set by a user, implicitly set
to a default value by the apiserver, or continuously adjusted by a Controller such as
and HorizontalPodAutoscaler.

{% method %}
### Last Applied Resource Config

When Apply creates or updates a Resource, it writes the Resource Config it Applied to an annotation on the
Resource.  This allows it to compare the last Resource Config it Applied to the current Resource
Config and identify fields that have been deleted.
{% sample lang="yaml" %}

```yaml
# deployment.yaml (Resource Config)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.7.9
```

```yaml
# Original Resource
Doesn't Exist
```

```yaml
# Applied Resource
kind: Deployment
metadata:
  annotations:
    # ...
    # This is the deployment.yaml Resource Config written as an annotation on the object
    # It was written by kubectl apply when the object was created
    kubectl.kubernetes.io/last-applied-configuration: |
      {"apiVersion":"apps/v1","kind":"Deployment",
      "metadata":{"annotations":{},"name":"nginx-deployment","namespace":"default"},
      "spec":{"selector":{"matchLabels":{"app":nginx}},"template":{"metadata":{"labels":{"app":"nginx"}},
      "spec":{"containers":[{"image":"nginx:1.7.9","name":"nginx"}]}}}}
  # ...
spec:
  # ...
status:
  # ...

```

{% endmethod %}

## Merging Resources

Following are the merge semantics for Resources:

{% method %}
**Adding Fields:**

- Fields present in the Resource Config that are missing from the Resource will be added to the
  Resource.
- Fields will be added to the Last Applied Resource Config
{% sample lang="yaml" %}

```yaml
# deployment.yaml (Resource Config)
apiVersion: apps/v1
kind: Deployment
metadata:
  # ...
  name: nginx-deployment
spec:
  # ...
  minReadySeconds: 3
```

```yaml
# Original Resource
kind: Deployment
metadata:
  # ...
  name: nginx-deployment
spec:
  # ...
status:
  # ...
```

```yaml
# Applied Resource
kind: Deployment
metadata:
  # ...
  name: nginx-deployment
spec:
  # ...
  minReadySeconds: 3
status:
  # ...
```
{% endmethod %}

{% method %}
**Updating Fields**

- Fields present in the Resource Config that are also present in the Resource will be merged recursively
  until a primitive field is updated, or a field is added / deleted.
- Fields will be updated in the Last Applied Resource Config
{% sample lang="yaml" %}

```yaml
# deployment.yaml (Resource Config)
apiVersion: apps/v1
kind: Deployment
metadata:
  # ...
  name: nginx-deployment
spec:
  # ...
  replicas: 2
```

```yaml
# Original Resource
kind: Deployment
metadata:
  # ...
  name: nginx-deployment
spec:
  # ...
  # could be defaulted or set by Resource Config
  replicas: 1
status:
  # ...
```

```yaml
# Applied Resource
kind: Deployment
metadata:
  # ...
  name: nginx-deployment
spec:
  # ...
  # updated
  replicas: 2
status:
  # ...
```
{% endmethod %}

{% method %}
**Deleting Fields**

- Fields present in the **Last Applied Resource Config** that have been removed from the Resource Config
  will be deleted from the Resource.
- Fields set to *null* in the Resource Config that are present in the Resource Config will be deleted from the
  Resource.
- Fields will be removed from the Last Applied Resource Config

{% sample lang="yaml" %}

```yaml
# deployment.yaml (Resource Config)
apiVersion: apps/v1
kind: Deployment
metadata:
  # ...
  name: nginx-deployment
spec:
  # ...
```

```yaml
# Original Resource
kind: Deployment
metadata:
  # ...
  name: nginx-deployment
  # Containers replicas and minReadySeconds
  kubectl.kubernetes.io/last-applied-configuration: |
      {"apiVersion":"apps/v1","kind":"Deployment", "spec":{"replicas": "2", "minReadySeconds": "3", ...}, "metadata": {...}}
spec:
  # ...
  minReadySeconds: 3
  replicas: 2
status:
  # ...
```

```yaml
# Applied Resource
kind: Deployment
metadata:
  # ...
  name: nginx-deployment
  kubectl.kubernetes.io/last-applied-configuration: |
      {"apiVersion":"apps/v1","kind":"Deployment", "spec":{...}, "metadata": {...}}
spec:
  # ...
  # deleted and then defaulted, but not in Last Applied
  replicas: 1
  # minReadySeconds deleted
status:
  # ...
```
{% endmethod %}

{% panel style="danger", title="Removing Fields from Resource Config" %}
Simply removing a field from the Resource Config will *not* transfer the ownership to the cluster.
Instead it will delete the field from the Resource.  If a field is set in the Resource Config and
the user wants to give up ownership (e.g. removing `replicas` from the Resource Config and using
and autoscaler), the user must first remove it from the last Applied Resource Config stored by the
cluster.

This can be performed using `kubectl apply edit-last-applied` to delete the `replicas` field from
the **Last Applied Resource Config**, and then deleting it from the **Resource Config.**
{% endpanel %}

## Field Merge Semantics

### Merging Primitives

Primitive fields are merged by replacing the current value with the new value.

**Field Creation:** Add the primitive field

**Field Update:** Change the primitive field value

**Field Deletion:** Delete the primitive field

| Field in Resource Config  | Field in Resource | Field in Last Applied | Action                                  |
|---------------------------|-------------------|-----------------------|-----------------------------------------|
| Yes                       | Yes               | -                     | Set live to the Resource Config value.  |
| Yes                       | No                | -                     | Set live to the Resource Config value.  |
| No                        | -                 | Yes                   | Remove from Resource.                   |
| No                        | -                 | No                    | Do nothing.                             |


### Merging Objects

Objects fields are updated by merging the sub-fields recursively (by field name) until a primitive field is found or
the field is added / deleted.

**Field Creation:** Add the object field

**Field Update:** Recursively compare object sub-field values and merge them

**Field Deletion:** Delete the object field

**Merge Table:** For each field merge Resource Config and Resource values with the same name

| Field in Resource Config  | Field in Resource | Field in Last Applied | Action                                    |
|---------------------------|-------------------|-----------------------|-------------------------------------------|
| Yes                       | Yes               | -                     | Recursively merge the Resource Config and Resource values.         |
| Yes                       | No                | -                     | Set live to the Resource Config value.    |
| No                        | -                 | Yes                   | Remove field from Resource.                     |
| No                        | -                 | No                    | Do nothing.                               |

### Merging Maps

Map fields are updated by merging the elements (by key) until a primitive field is found or the value is
added / deleted.

**Field Creation:** Add the map field

**Field Update:** Recursively compare map values by key and merge them

**Field Deletion:** Delete the map field

**Merge Table:** For each map element merge Resource Config and Resource values with the same key

| Key in Resource Config    | Key   in Resource | Key in Last Applied   | Action                                    |
|---------------------------|-------------------|-----------------------|-------------------------------------------|
| Yes                       | Yes               | -                     | Recursively merge the Resource Config and Resource values.        |
| Yes                       | No                | -                     | Set live to the Resource Config value.    |
| No                        | -                 | Yes                   | Remove map element from Resource.                     |
| No                        | -                 | No                    | Do nothing.                               |

### Merging Lists of Primitives

Lists of primitives will be merged if they have a `patch strategy: merge` on the field otherwise they will
be replaced.  [Finalizer list example](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.12/#objectmeta-v1-meta)

**Merge Strategy:**

- Merged primitive lists behave like ordered sets
- Replace primitive lists are replaced when merged

**Ordering:** Uses the ordering specified in the Resource Config.  Elements not specified in the Resource Config
do not have ordering guarantees with respect to the elements in the Resource Config.

**Merge Table:** For each list element merge Resource Config and Resource element with the same value

| Element in Resource Config  | Element in Resource | Element in Last Applied | Action                                  |
|---------------------------|-------------------|-----------------------|-----------------------------------------|
| Yes                       | Yes               | -                     | Do nothing  |
| Yes                       | No                | -                     | Add to list.  |
| No                        | -                 | Yes                   | Remove from list.                   |
| No                        | -                 | No                    | Do nothing.                             |

{% method %}

This merge strategy uses the patch merge key to identify container elements in a list and merge them.
The `patch merge key` is defined in the [Kubernetes API](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.12/#podspec-v1-core)
on the field.

{% sample lang="yaml" %}

```yaml
# Last Applied
args: ["a", "b"]
```

```yaml
# Resource Config (Local)
args: ["a", "c"]
```

```yaml
# Resource (Live)
args: ["a", "b", "d"]
```

```yaml
# Applied Resource
args: ["a", "c", "d"]
```

{% endmethod %}

### Merging Lists of Objects

**Merge Strategy:** Lists of primitives may be merged or replaced.  Lists are merged if the list has a `patch strategy` of *merge*
and a `patch merge key` on the list field.  [Container list example](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.12/#podspec-v1-core).

**Merge Key:** The `patch merge key` is used to identify same elements in a list.  Unlike map elements (keyed by key) and object fields
(keyed by field name), lists don't have a built-in merge identity for elements (index does not define identity).
Instead an object field is used as a synthetic *key/value* for merging elements.  This fields is the
`patch merge key`.  List elements with the same patch merge key will be merged when lists are merged.

**Ordering:** Uses the ordering specified in the Resource Config.  Elements not specified in the Resource Config
do not have ordering guarantees.

**Merge Table:** For each list element merge Resource Config and Resource element where the elements have the same
value for the `patch merge key`

| Element in Resource Config  | Element in Resource | Element in Last Applied | Action                                  |
|---------------------------|-------------------|-----------------------|-----------------------------------------|
| Yes                       | -               | -                       | Recursively merge the Resource Config and Resource values.  |
| Yes                       | No                | -                     | Add to list.  |
| No                        | -                 | Yes                   | Remove from list.                   |
| No                        | -                 | No                    | Do nothing.                             |

{% method %}

This merge strategy uses the patch merge key to identify container elements in a list and merge them.
The `patch merge key` is defined in the [Kubernetes API](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.12/#podspec-v1-core)
on the field.

{% sample lang="yaml" %}

```yaml
# Last Applied Resource Config
containers:
- name: nginx          # key: nginx
  image: nginx:1.10
- name: nginx-helper-a # key: nginx-helper-a; will be deleted in result
  image: helper:1.3
- name: nginx-helper-b # key: nginx-helper-b; will be retained
  image: helper:1.3
```

```yaml
# Resource Config (Local)
containers:
- name: nginx
  image: nginx:1.10
- name: nginx-helper-b
  image: helper:1.3
- name: nginx-helper-c # key: nginx-helper-c; will be added in result
  image: helper:1.3
```

```yaml
# Resource (Live)
containers:
- name: nginx
  image: nginx:1.10
- name: nginx-helper-a
  image: helper:1.3
- name: nginx-helper-b
  image: helper:1.3
  args: ["run"] # Field will be retained
- name: nginx-helper-d # key: nginx-helper-d; will be retained
  image: helper:1.3
```

```yaml
# Applied Resource
containers:
- name: nginx
  image: nginx:1.10
  # Element nginx-helper-a was Deleted
- name: nginx-helper-b
  image: helper:1.3
  # Field was Ignored
  args: ["run"]
  # Element was Added
- name: nginx-helper-c
  image: helper:1.3
  # Element was Ignored
- name: nginx-helper-d
  image: helper:1.3
```
{% endmethod %}

{% panel style="info", title="Edit and Set" %}
While `kubectl edit` and `kubectl set` ignore the Last Applied Resource Config, Apply will
change any values in the Resource Config set by either `kubectl edit` or `kubectl set`.
To ignore values set by `kubectl edit` or `kubectl set`:

- Use `kubectl apply edit-last-applied` to remove the value from the Last Applied (if it is present)
- Remove the field from the Resource Config

This is the same technique for retaining values set by cluster components such as autoscalers.
{% endpanel %}
