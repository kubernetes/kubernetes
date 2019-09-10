{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/CLQBQHR)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Set Labels for all Resources declared within a Project with `commonLabels`
- Set Annotations for all Resources declared within a Project with `commonAnnotations`
{% endpanel %}

# Setting Labels and Annotations

## Motivation

Users may want to define a common set of labels or annotations for all the Resource in a project.

- Identify the Resources within a project by querying their labels.
- Set metadata for all Resources within a project (e.g. environment=test).
- Copy or Fork an existing Project and add or change labels and annotations.

See [Bases and Variations](../app_customization/bases_and_variants.md) for more details on Copying Projects.

{% panel style="info", title="Reference" %}
- [commonLabels](../reference/kustomize.md#commonlabels)
- [commonAnnotations](../reference/kustomize.md#commonannotations)
{% endpanel %}


## Setting Labels for all Resources

{% method %}
**Example:** Add the labels declared in `commonLabels` to all Resources in the project.

**Important:** Once set, commonLabels should not be changed so as not to change the Selectors for Services
or Workloads.

{% sample lang="yaml" %}
**Input:** The kustomization.yaml and deployment.yaml files

```yaml
# kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
commonLabels:
  app: foo
  environment: test
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
    bar: baz
spec:
  selector:
    matchLabels:
      app: nginx
      bar: baz
  template:
    metadata:
      labels:
        app: nginx
        bar: baz
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
    app: foo # Label was changed
    environment: test # Label was added
    bar: baz # Label was ignored
  name: nginx-deployment
spec:
  selector:
    matchLabels:
      app: foo # Selector was changed
      environment: test # Selector was added
      bar: baz # Selector was ignored
  template:
    metadata:
      labels:
        app: foo # Label was changed
        environment: test # Label was added
        bar: baz # Label was ignored
    spec:
      containers:
      - image: nginx
        name: nginx
```
{% endmethod %}

{% panel style="warning", title="Propagating Labels to Selectors" %}
In addition to updating the labels for each Resource, any selectors will also be updated to target the
labels.  e.g. the selectors for Services in the project will be updated to include the commonLabels
*in addition* to the other labels.

**Note:** Once set, commonLabels should not be changed so as not to change the Selectors for Services
or Workloads.
{% endpanel %}

{% panel style="success", title="Common Labels" %}
The k8s.io documentation defines a set of [Common Labeling Conventions](https://kubernetes.io/docs/concepts/overview/working-with-objects/common-labels/)
that may be applied to Applications.

**Note:** commonLabels should only be set for **immutable** labels, since they will be applied to Selectors.

Labeling Workload Resources makes it simpler to query Pods - e.g. for the purpose of getting their logs.
{% endpanel %}


## Setting Annotations for all Resources

{% method %}
**Example:** Add the annotations declared in `commonAnnotations` to all Resources in the project.

{% sample lang="yaml" %}
**Input:** The kustomization.yaml and deployment.yaml files

```yaml
# kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
commonAnnotations:
  oncallPager: 800-555-1212
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
  # Annotation added to the Deployment
  annotations:
    oncallPager: 800-555-1212
  labels:
    app: nginx
  name: nginx-deployment
spec:
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      # Annotation also added to PodTemplate
      annotations:
        oncallPager: 800-555-1212
      labels:
        app: nginx
    spec:
      containers:
      - image: nginx
        name: nginx
```
{% endmethod %}

{% panel style="info", title="Propagating Annotations" %}
In addition to updating the annotations for each Resource, any fields that contain ObjectMeta
(e.g. PodTemplate) will also have the annotations added.
{% endpanel %}
