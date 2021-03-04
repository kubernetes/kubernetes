{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/CLQBQHR)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Set the Namespace for all Resources within a Project with `namespace`
- Prefix the Names of all Resources within a Project with `namePrefix`
- Suffix the Names of all Resources within a Project with `nameSuffix`
{% endpanel %}

# Setting Namespaces and Names

## Motivation

It may be useful to enforce consistency across the namespace and names of all Resources within
a Project.

- Ensure all Resources are in the correct Namespace
- Ensure all Resources share a common naming convention
- Copy or Fork an existing Project and change the Namespace / Names

See [Bases and Variations](../app_customization/bases_and_variants.md) for more details on Copying Projects.


{% panel style="info", title="Reference" %}
- [namespace](../reference/kustomize.md#namespace)
- [namePrefix](../reference/kustomize.md#nameprefix)
- [nameSuffix](../reference/kustomize.md#namesuffix)
{% endpanel %}


## Setting the Namespace for all Resources

Reference: 

The Namespace for all namespaced Resources declared in the Resource Config may be set with `namespace`.
This sets the namespace for both generated Resources (e.g. ConfigMaps and Secrets) and non-generated
Resources.

{% method %}
**Example:** Set the `namespace` specified in the `kustomization.yaml` on the namespaced Resources.

{% sample lang="yaml" %}
**Input:** The kustomization.yaml and deployment.yaml files

```yaml
# kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: my-namespace
resources:
- deployment.yaml
```

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
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
        image: nginx
```

**Applied:** The Resource that is Applied to the cluster

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: nginx
  name: nginx-deployment
  # The namespace has been added
  namespace: my-namespace
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
      - image: nginx
        name: nginx
```

{% endmethod %}

{% panel style="info", title="Overriding Namespaces" %}
Setting the namespace will override the namespace on Resources if it is already set.
{% endpanel %}

## Setting a Name prefix or suffix for all Resources

A name prefix or suffix can be set for all resources using `namePrefix` or
`nameSuffix`.

{% method %}
**Example:** Prefix the names of all Resources.

{% sample lang="yaml" %}
**Input:** The kustomization.yaml and deployment.yaml files

```yaml
# kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namePrefix: foo-
resources:
- deployment.yaml
```

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
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
        image: nginx
```

**Applied:** The Resource that is Applied to the cluster

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: nginx
  # The name has been prefixed with "foo-"
  name: foo-nginx-deployment
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
      - image: nginx
        name: nginx
```
{% endmethod %}

{% panel style="info", title="Propagation of the Name to Object References" %}
Resources such as Deployments and StatefulSets may reference other Resources such as
ConfigMaps and Secrets in the Pod Spec.

This sets a name prefix or suffix for both generated Resources (e.g. ConfigMaps 
and Secrets) and non-generated Resources.

The namePrefix or nameSuffix that is applied is propagated to references to updated resources -
e.g. references to Secrets and ConfigMaps are updated with the namePrefix and nameSuffix.
{% endpanel %}

{% method %}
**Example:** Prefix the names of all Resources.

This will update the ConfigMap reference in the Deployment to have the `foo` prefix.

{% sample lang="yaml" %}
**Input:** The kustomization.yaml and deployment.yaml files

```yaml
# kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namePrefix: foo-
configMapGenerator:
- name: props
  literals:	
  - BAR=baz
resources:
- deployment.yaml
```

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
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
        image: nginx
        env:
        - name: BAR
          valueFrom:
            configMapKeyRef:
              name: props
              key: BAR
```

**Applied:** The Resource that is Applied to the cluster

 ```yaml
apiVersion: v1
data:
  BAR: baz
kind: ConfigMap
metadata:
  creationTimestamp: null
  name: foo-props-44kfh86dgg
---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: nginx
  name: foo-nginx-deployment
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
      - env:
        - name: BAR
          valueFrom:
            configMapKeyRef:
              key: BAR
              name: foo-props-44kfh86dgg
        image: nginx
        name: nginx
```
{% endmethod %}

{% panel style="info", title="References" %}
Apply will propagate the `namePrefix` to any place Resources within the project are referenced by other Resources
including:

- Service references from StatefulSets
- ConfigMap references from PodSpecs
- Secret references from PodSpecs
{% endpanel %}