{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/CLQBQHR)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Apply Creates and Updates Resources in a cluster through running `kubectl apply` on Resource Config.
- Apply manages complexity such as ordering of operations and merging user defined and cluster defined state.
{% endpanel %}

# Apply

## Motivation

Apply is a command that will update a Kubernetes cluster to match state defined locally in files.

```bash
kubectl apply
```

- Fully declarative - don't need to specify create or update - just manage files
- Merges user owned state (e.g. Service `selector`) with state owned by the cluster (e.g. Service `clusterIp`)

## Definitions

- **Resources**: *Objects* in a cluster - e.g. Deployments, Services, etc.
- **Resource Config**: *Files* declaring the desired state for Resources - e.g. deployment.yaml.
  Resources are created and updated using Apply with these files.

*kubectl apply* Creates and Updates Resources through local or remote files.  This may be through
either raw Resource Config or *kustomization.yaml*.

## Usage

{% method %}

Though Apply can be run directly against Resource Config files or directories using `-f`, it is recommended
to run Apply against a `kustomization.yaml` using `-k`.  The `kustomization.yaml` allows users to define
configuration that cuts across many Resources (e.g. namespace).

{% sample lang="yaml" %}

```yaml
# kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

# list of Resource Config to be Applied
resources:
- deployment.yaml

# namespace to deploy all Resources to
namespace: default

# labels added to all Resources
commonLabels:
  app: example
  env: test
```

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
  labels:
    component: nginx
    tier: frontend
spec:
  selector:
    matchLabels:
      component: nginx
      tier: frontend
  template:
    metadata:
      labels:
        component: nginx
        tier: frontend
    spec:
      containers:
      - name: nginx
        image: nginx:1.15.4
```

{% endmethod %}

{% method %}
Users run Apply on directories containing `kustomization.yaml` files using `-k` or on raw
ResourceConfig files using `-f`.

{% sample lang="yaml" %}
```bash
# Apply the Resource Config
kubectl apply -k .

# View the Resources
kubectl get -k .
```
{% endmethod %}

{% panel style="info", title="Multi-Resource Configs" %}
A single Resource Config file may declare multiple Resources separated by `\n---\n`.
{% endpanel %}

## CRUD Operations

### Creating Resources

Any Resources that do not exist and are declared in Resource Config when Apply is run will be Created.

### Updating Resources

Any Resources that already exist and are declared in Resource Config when Apply is run may be Updated.

**Added Fields**

Any fields that have been added to the Resource Config will be set on the Resource.

**Updated Fields** 
 
Any fields that contain different values for the fields specified locally in the Resource Config from what is
in the Resource will be updated by merging the Resource Config into the live Resource.  See [merging](field_merge_semantics.md)
for more details.

**Deleted Fields**

Fields that were in the Resource Config the last time Apply was run, will be deleted from the Resource, and
return to their default values.

**Unmanaged Fields**

Fields that were not specified in the Resource Config but are set on the Resource will be left unmodified.

### Deleting Resources

Declarative deletion of Resources does not yet exist in a usable form, but is under development.

{% panel style="info", title="Continuously Applying The Hard Way" %}
In some cases, it may be useful to automatically Apply changes when ever the Resource Config is changed.

This example uses the unix `watch` command to periodically invoke Apply against a target.
`watch -n 60 kubectl apply -k https://github.com/myorg/myrepo`

{% endpanel %}

## Resource Creation Ordering

Certain Resource Types may be dependent on other Resource Types being created first.  e.g. Namespaced
Resources on the Namespaces, RoleBindings on Roles, CustomResources on the CRDs, etc.

When used with a `kustomization.yaml`, Apply sorts the Resources by Resource type to ensure Resources
with these dependencies are created in the correct order.
