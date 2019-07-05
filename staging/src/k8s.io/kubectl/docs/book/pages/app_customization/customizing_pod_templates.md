{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/C855WZW)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Override Base Pod and PodTemplate Image **Names** and **Tags**
- Override Base Pod and PodTemplate Environment Variables and Arguments
{% endpanel %}

# Customizing Pods

## Motivation

It is common for users to customize their Applications for specific environments.
Simple customizations to Pod Templates may be through **Images, Environment Variables and
Command Line Arguments**.

Common examples include:

- Running **different versions of an Image** for dev, test, canary, production
- Configuring **different Pod Environment Variables and Arguments** for dev, test, canary, production

{% panel style="info", title="Reference" %}
- [images](../reference/kustomize.md#images)
- [configMapGenerator](../reference/kustomize.md#configmapgenerator)
- [secretGenerator](../reference/kustomize.md#secretgenerator)
{% endpanel %}

## Customizing Images

{% method %}
**Use Case:** Different Environments (test, dev, staging, canary, prod) use images with different tags.

Override the name or tag for an `image` field from a [Pod Template](https://kubernetes.io/docs/concepts/workloads/pods/pod-overview/#pod-templates)
in a base by specifying the `images` field in the `kustomization.yaml`.

| Field     | Description                                                              | Example Field | Example Result |
|-----------|--------------------------------------------------------------------------|----------| --- |
| `name`    | Match images with this image name| `name: nginx`| |
| `newTag`  | Override the image **tag** or **digest** for images whose image name matches `name`    | `newTag: new` | `nginx:old` -> `nginx:new` |
| `newName` | Override the image **name** for images whose image name matches `name`   | `newImage: nginx-special` | `nginx:old` -> `nginx-special:old` |

{% sample lang="yaml" %}
**Input:** The `kustomization.yaml` file

```yaml
# kustomization.yaml
bases:
- ../base
images:
  - name: nginx-pod
    newTag: 1.15
    newName: nginx-pod-2
```

**Base:** Resources to be modified by the `kustomization.yaml`

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
        image: nginx-pod
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
      # The image image tag has been changed for the container
      - name: nginx
        image: nginx-pod-2:1.15
```
{% endmethod %}


{% panel style="info", title="Replacing Images" %}
`newImage` allows an image name to be replaced with another arbitrary image name.  e.g. you could
call your image `webserver` or `database` and replace it with `nginx` or `mysql`.

For more information on customizing images, see [Container Images](../app_management/container_images.md).
{% endpanel %}

## Customizing Pod Environment Variables

{% method %}

**Use Case:** Different Environments (test, dev, staging, canary, prod) are configured with
different Environment Variables.

Override Pod Environment Variables.

- Base uses ConfigMap data in Pods as Environment Variables
- Each Variant overrides or extends ConfigMap data

{% sample lang="yaml" %}
**Input:** The kustomization.yaml file

```yaml
# kustomization.yaml
bases:
- ../base
configMapGenerator:
- name: special-config
  behavior: merge
  literals:
  - special.how=very # override the base value
  - special.type=charm # add a value to the base
```

**Base: kustomization.yaml and Resources**

```yaml
# ../base/kustomization.yaml
resources:
- deployment.yaml
configMapGenerator:
- name: special-config
  behavior: merge
  literals:
  - special.how=some # this value is overridden
  - special.other=that # this value is added
```

```yaml
# ../base/deployment.yaml
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
        envFrom:
        - configMapRef:
            name: special-config
```

**Applied:** The Resources that are Applied to the cluster

```yaml
# Generated Variant Resource
apiVersion: v1
kind: ConfigMap
metadata:
  name: special-config-82tc88cmcg
data:
  special.how: very
  special.type: charm
  special.other: that
---
# Unmodified Base Resource
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
      - name: nginx
        image: nginx
        envFrom:
        # Container env will have the overridden ConfigMap values
        - configMapRef:
            name: special-config-82tc88cmcg
```
{% endmethod %}

See [ConfigMaps and Secrets](../app_management/secrets_and_configmaps.md).


## Customizing Pod Command Arguments

{% method %}
**Use Case:** Different Environments (test, dev, staging, canary, prod) provide different Commandline
Arguments to a Pod.

Override Pod Command Arguments.

- Base uses ConfigMap data in Pods as Command Arguments
- Each Variant defines different ConfigMap data

{% sample lang="yaml" %}
**Input:** The kustomization.yaml file

```yaml
# kustomization.yaml
bases:
- ../base
configMapGenerator:
- name: special-config
  behavior: merge
  literals:
  - SPECIAL_LEVEL=very
  - SPECIAL_TYPE=charm
```

```yaml
# ../base/kustomization.yaml
resources:
- deployment.yaml

configMapGenerator:
- name: special-config
  literals:
  - SPECIAL_LEVEL=override.me
  - SPECIAL_TYPE=override.me
```

```yaml
# ../base/deployment.yaml
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
      - name: test-container
        image: k8s.gcr.io/busybox
        command: [ "/bin/sh" ]
        # Use the ConfigMap Environment Variables in the Command
        args: ["-c", "echo $(SPECIAL_LEVEL_KEY) $(SPECIAL_TYPE_KEY)" ]
        env:
          - name: SPECIAL_LEVEL_KEY
            valueFrom:
              configMapKeyRef:
                name: special-config
                key: SPECIAL_LEVEL
          - name: SPECIAL_TYPE_KEY
            valueFrom:
              configMapKeyRef:
                name: special-config
                key: SPECIAL_TYPE
```

**Applied:** The Resources that are Applied to the cluster

```yaml
# Generated Variant Resource
apiVersion: v1
kind: ConfigMap
metadata:
  name: special-config-82tc88cmcg
data:
  SPECIAL_LEVEL: very
  SPECIAL_TYPE: charm
---
# Unmodified Base Resource
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
      - image: k8s.gcr.io/busybox
        name: test-container
        command:
        - /bin/sh
        args:
        - -c
        # Container args will have the overridden ConfigMap values
        - echo $(SPECIAL_LEVEL_KEY) $(SPECIAL_TYPE_KEY)
        env:
        - name: SPECIAL_LEVEL_KEY
          valueFrom:
            configMapKeyRef:
              key: SPECIAL_LEVEL
              name: special-config-82tc88cmcg
        - name: SPECIAL_TYPE_KEY
          valueFrom:
            configMapKeyRef:
              key: SPECIAL_TYPE
              name: special-config-82tc88cmcg
```

{% endmethod %}

{% panel style="info", title="More Info" %}
See [Secrets and ConfigMaps](../app_management/secrets_and_configmaps.md) for more information on ConfigMap and Secret generation.
{% endpanel %}

