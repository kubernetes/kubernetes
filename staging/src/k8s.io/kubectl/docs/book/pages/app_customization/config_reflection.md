{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/C855WZW)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Inject the values of other Resource Config fields into Pod Env Vars and Command Args with `vars`.
{% endpanel %}

# Config Reflection

## Motivation

Applications running in Pods may need to know about Application context or configuration.
For example, a **Pod may take the name of Service defined in the Project as a command argument**.
Instead of hard coding the value of the Service directly into the PodSpec, users can **reference
the Service value using a `vars` entry**.  If the value is updated or transformed by the
`kustomization.yaml` file (e.g. by setting a `namePrefix`), the value will be propagated
to where it is referenced in the PodSpec.

{% panel style="info", title="Reference" %}
 - [vars](../reference/kustomize.md#var)
 {% endpanel %} 

## Vars

The `vars` section contains variable references to Resource Config fields within the project.  They require
the following to be defined:

- Resource Kind
- Resource Version
- Resource name
- Field path

{% method %}

**Example:** Set the Pod command argument to the value of a Service name.

Apply will resolve `$(BACKEND_SERVICE_NAME)` to a value using the object reference
specified in `vars`.

{% sample lang="yaml" %}
**Input:** The kustomization.yaml, deployment.yaml and service.yaml files

```yaml
# kustomization.yaml
namePrefix: "test-"
vars:
  # Name of the variable so it can be referenced
- name: BACKEND_SERVICE_NAME
  # GVK of the object with the field
  objref:
    kind: Service
    name: backend-service
    apiVersion: v1
  # Path to the field
  fieldref:
    fieldpath: metadata.name
resources:
- deployment.yaml
- service.yaml
```

```yaml
# service.yaml
kind: Service
apiVersion: v1
metadata:
  # Value of the variable.  This will be customized with
  # a namePrefix, and change the Variable value.
  name: backend-service
spec:
  selector:
    app: backend
  ports:
  - protocol: TCP
    port: 80
    targetPort: 9376
```

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: curl-deployment
  labels:
    app: curl
spec:
  selector:
    matchLabels:
      app: curl
  template:
    metadata:
      labels:
        app: curl
    spec:
      containers:
      - name: curl
        image: ubuntu
        # Reference the Service name field value as a variable
        command: ["curl", "$(BACKEND_SERVICE_NAME)"]
```

**Applied:** The Resources that are Applied to the cluster

```yaml
apiVersion: v1
kind: Service
metadata:
  name: test-backend-service
spec:
  ports:
  - port: 80
    protocol: TCP
    targetPort: 9376
  selector:
    app: backend
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-curl-deployment
  labels:
    app: curl
spec:
  selector:
    matchLabels:
      app: curl
  template:
    metadata:
      labels:
        app: curl
    spec:
      containers:
      - command:
        - curl
        # $(BACKEND_SERVICE_NAME) has been resolved to
        # test-backend-service
        - test-backend-service
        image: ubuntu
        name: curl
```
{% endmethod %}

{% panel style="warning", title="Referencing Variables" %}
Variables are intended only to inject Resource Config into Pods.  They are
**not** intended as a general templating mechanism.  Overriding values should be done with
patches instead of variables.  See [Bases and Variations](bases_and_variants.md).
{% endpanel %}
