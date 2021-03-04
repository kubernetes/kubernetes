{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/C855WZW)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Customize arbitrary fields from arbitrary Resources in a Base.
{% endpanel %}

# Customizing Resource Fields

## Motivation

It is often necessary for users to want to **modify arbitrary fields** from a Base, such
as resource reservations for Pods, replicas on Deployments, etc.  Overlays and patches can
be used by Variants to specify fields values which will override the Base field values.

{% panel style="info", title="Reference" %}
- [patchesjson6902](../reference/kustomize.md#patchesjson6902)
- [patchesStrategicMerge](../reference/kustomize.md#patchesstrategicmerge)
{% endpanel %}

## Customizing Arbitrary Fields with Overlays

{% method %}
Arbitrary **fields may be added, changed, or deleted** by supplying *Overlays* against the
Resources provided by the Base.  **Overlays are sparse Resource definitions** that
allow arbitrary customizations to be performed without requiring a base to expose
the customization as a template.

Overlays require the *Group, Version, Kind* and *Name* of the Resource to be specified, as
well as any fields that should be set on the base Resource.  Overlays are applied using
*StrategicMergePatch*.

**Use Case:** Different Environments (test, dev, staging, canary, prod) require fields such as
replicas or resources to be overridden.

{% sample lang="yaml" %}
**Input:** The kustomization.yaml file and overlay

```yaml
# kustomization.yaml
bases:
- ../base
patchesStrategicMerge:
- overlay.yaml
```

```yaml
# overlay.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  # override replicas
  replicas: 3
  template:
    spec:
      containers:
      - name: nginx
        # override resources
        resources:
          limits:
            cpu: "1"
          requests:
            cpu: "0.5"
```

**Base:**

```yaml
# ../base/kustomization.yaml
resources:
- deployment.yaml
```

```yaml
# ../base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: nginx
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
      - image: nginx
        name: nginx
        resources:
          limits:
            cpu: "0.2"
          requests:
            cpu: "0.1"
```

**Applied:** The Resource that is Applied to the cluster

```yaml
# Overlayed Base Resource
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: nginx
  name: nginx-deployment
spec:
  # replicas field has been added
  replicas: 3
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
        # resources have been overridden
        resources:
          limits:
            cpu: "1"
          requests:
            cpu: "0.5"
```

{% endmethod %}

{% panel style="info", title="Merge Semantics for Overlays" %}
Overlays use the same [merge semantics](../app_management/field_merge_semantics.md) as Applying Resource Config to cluster.  One difference
is that there is no *Last Applied Resource Config* when merging overlays, so fields may only be deleted
if they are explicitly set to nil.
{% endpanel %}

## Customizing Arbitrary Fields with JsonPatch

{% method %}
Arbitrary fields may be added, changed, or deleted by supplying *JSON Patches* against the
Resources provided by the base.

**Use Case:** Different Environments (test, dev, staging, canary, prod) require fields such as
replicas or resources to be overridden.

JSON Patches are [RFC 6902](https://tools.ietf.org/html/rfc6902) patches that are applied
to resources.  Patches require the *Group, Version, Kind* and *Name* of the Resource to be
specified in addition to the Patch.  Patches offer a number of powerful imperative operations
for modifying the base Resources.

{% sample lang="yaml" %}
**Input:** The kustomization.yaml file

```yaml
# kustomization.yaml
bases:
- ../base
patchesJson6902:
- target:
    group: apps
    version: v1
    kind: Deployment
    name: nginx-deployment
  path: patch.yaml
```

```yaml
# patch.yaml
- op: add
  path: /spec/replicas
  value: 3
```

```yaml
# ../base/kustomization.yaml
resources:
- deployment.yaml
```

```yaml
# ../base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: nginx
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
      - image: nginx
        name: nginx
```

**Applied:** The Resource that is Applied to the cluster

```yaml
# Patched Base Resource
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: nginx
  name: nginx-deployment
spec:
  # replicas field has been added
  replicas: 3
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
