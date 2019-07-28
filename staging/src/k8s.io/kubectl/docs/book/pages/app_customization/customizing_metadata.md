{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/C855WZW)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Customize Base Resource Namespaces
- Customize Base Resource Names with Prefixes or Suffixes
- Customize Base Resource Labels or Annotations
{% endpanel %}

# Customizing Resource Metadata

## Motivation

It is common for users to customize the metadata of their Applications - including
the **names, namespaces, labels and annotations**.

Examples:

- Overriding the Namespace
- Overriding the Names of Resources by supplying a Prefix or Suffix
- Overriding Labels and Annotations
- Running **multiple instances of the same White-Box Base** using the above techniques
 
 {% panel style="info", title="Reference" %}
 - [namespace](../reference/kustomize.md#namespace)
 - [namePrefix](../reference/kustomize.md#nameprefix)
 - [nameSuffix](../reference/kustomize.md#namesuffix)
 {% endpanel %}
 
## Customizing Resource Namespaces

{% method %}
**Use Case:**
- Change the Namespace for Resources from Base.

Customize the Namespace of all Resources in the Base by adding `namespace`.

{% sample lang="yaml" %}
**Input:** The kustomization.yaml file

```yaml
# kustomization.yaml
bases:
- ../base
namespace: test
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
  namespace: default
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
# Modified Base Resource
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: nginx
  name: nginx-deployment
  # Namepace has been changed to test
  namespace: test
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
 
## Customizing Resource Name Prefixes and Suffixes

{% method %}
**Use Case:**
- Run multiple instances of the same Base.
- Create naming conventions for different Environments (test, dev, staging, canary, prod).

Customize the Name of all Resources in the Base by adding `namePrefix` or `nameSuffix` in Variants.

{% sample lang="yaml" %}
**Input:** The kustomization.yaml file

```yaml
# kustomization.yaml
bases:
- ../base
namePrefix: test-
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
```

**Applied:** The Resource that is Applied to the cluster

```yaml
# Modified Base Resource
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: nginx
  # Name has been prefixed with the environment
  name: test-nginx-deployment
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

See [Namespaces and Names](../app_management/namespaces_and_names.md).

{% panel style="success", title="Chaining Name Prefixes" %}
Name Prefix's and Suffix's in Bases will be concatenated with Name Prefix's
and Suffix's specified in Variants - e.g. if a Base has a Name Prefix of `app-name-`
and the Variant has a Name Prefix of `test-` the Applied Resources will have
a Name Prefix of `test-app-name-`.
{% endpanel %}

## Customizing Resource Labels and Annotations

{% method %}
**Use Case:**
- Create Label or Annotation conventions for different Environments (test, dev, staging, canary, prod).

Customize the Labels and Annotations of all Resources in the Base by adding a
`commonLabels` or `commonAnnotations` in the variants.

{% sample lang="yaml" %}
**Input:** The kustomization.yaml file

```yaml
# kustomization.yaml
bases:
- ../base
commonLabels:
  app: test-nginx
  environment: test
commonAnnotations:
  oncallPager: 800-555-1212
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
    base: label
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
# Modified Base Resource
apiVersion: apps/v1
kind: Deployment
metadata:
  # labels have been overridden
  labels:
    app: test-nginx
    environment: test
    base: label
  # annotations have been overridden
  annotations:
    oncallPager: 800-555-1212
  name: nginx-deployment
spec:
  selector:
    matchLabels:
      app: test-nginx
      environment: test
      base: label
  template:
    metadata:
      labels:
       app: test-nginx
       environment: test
       base: label
    spec:
      containers:
      - image: nginx
        name: nginx
```

{% endmethod %}