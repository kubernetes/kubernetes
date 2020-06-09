{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/JH35X82)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Imperatively Create a Resources
{% endpanel %}

# Creating Resources

## Motivation

Create Resources directly from the command line for the purposes of development or debugging.
Not for production Application Management.

{% method %}
## Deployment

A Deployment can be created with the `create deployment` command.

{% sample lang="yaml" %}

```bash
kubectl create deployment my-dep --image=busybox
```

{% endmethod %}

{% panel style="success", title="Running and Attaching" %}
It is possible to run a container and immediately attach to it using the `-i -t` flags.  e.g.
`kubectl run -t -i my-dep --image ubuntu -- bash`
{% endpanel %}

{% method %}
## ConfigMap

Create a configmap based on a file, directory, or specified literal value.

A single configmap may package one or more key/value pairs.

When creating a configmap based on a file, the key will default to the basename of the file, and the value will default
to the file content.  If the basename is an invalid key, you may specify an alternate key.

When creating a configmap based on a directory, each file whose basename is a valid key in the directory will be
packaged into the configmap.  Any directory entries except regular files are ignored (e.g. subdirectories, symlinks,
devices, pipes, etc).

{% sample lang="yaml" %}

```bash
# Create a new configmap named my-config based on folder bar
kubectl create configmap my-config --from-file=path/to/bar
```

```bash
# Create a new configmap named my-config with specified keys instead of file basenames on disk
kubectl create configmap my-config --from-file=key1=/path/to/bar/file1.txt --from-file=key2=/path/to/bar/file2.txt
  ```

```bash
# Create a new configmap named my-config with key1=config1 and key2=config2
kubectl create configmap my-config --from-literal=key1=config1 --from-literal=key2=config2
```

```bash
# Create a new configmap named my-config from an env file
kubectl create configmap my-config --from-env-file=path/to/bar.env
```

{% endmethod %}

{% method %}
## Secret

Create a new secret named my-secret with keys for each file in folder bar

{% sample lang="yaml" %}

```bash
kubectl create secret generic my-secret --from-file=path/to/bar
```

{% endmethod %}

{% panel style="success", title="Bootstrapping Config" %}
Imperative commands can be used to bootstrap config by using `--dry-run=client -o yaml`.
`kubectl create secret generic my-secret --from-file=path/to/bar --dry-run=client -o yaml`
{% endpanel %}

{% method %}
## Namespace

Create a new namespace named my-namespace

{% sample lang="yaml" %}

```bash
kubectl create namespace my-namespace
```

{% endmethod %}

## Auth Resources

{% method %}
### ClusterRole

Create a ClusterRole named "foo" with API Group specified.

{% sample lang="yaml" %}

```bash
kubectl create clusterrole foo --verb=get,list,watch --resource=rs.extensions
```

{% endmethod %}

{% method %}
### ClusterRoleBinding

Create a role binding to give a user cluster admin permissions.

{% sample lang="yaml" %}

```bash
kubectl create clusterrolebinding <choose-a-name> --clusterrole=cluster-admin --user=<your-cloud-email-account>
```

{% endmethod %}

{% panel style="info", title="Required Admin Permissions" %}
The cluster-admin role maybe required for creating new RBAC bindings.
{% endpanel %}

{% method %}
### Role

Create a Role named "foo" with API Group specified.

{% sample lang="yaml" %}

```bash
kubectl create role foo --verb=get,list,watch --resource=rs.extensions
```

{% endmethod %}

{% method %}
### RoleBinding

Create a RoleBinding for user1, user2, and group1 using the admin ClusterRole.

{% sample lang="yaml" %}

```bash
kubectl create rolebinding admin --clusterrole=admin --user=user1 --user=user2 --group=group1
```

{% endmethod %}

{% method %}
### ServiceAccount

Create a new service account named my-service-account

{% sample lang="yaml" %}

```bash
kubectl create serviceaccount my-service-account
```

{% endmethod %}
