{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/JH35X82)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Copy files to and from Containers in a cluster
{% endpanel %}

# Copying Container Files

## Motivation

- Copying files from Containers in a cluster to a local filesystem
- Copying files from a local filesystem to Containers in a cluster

{% panel style="warning", title="Install Tar" %}
Copy requires that *tar* be installed in the container image.
{% endpanel %}

{% method %}
## Local to Remote

Copy a local file to a remote Pod in a cluster.

- Local file format is `<path>`
- Remote file format is `<pod-name>:<path>`

{% sample lang="yaml" %}

```bash
kubectl cp /tmp/foo_dir <some-pod>:/tmp/bar_dir
```

{% endmethod %}

{% method %}
## Remote to Local

Copy a remote file from a Pod to a local file.

- Local file format is `<path>`
- Remote file format is `<pod-name>:<path>`

{% sample lang="yaml" %}

```bash
kubectl cp <some-pod>:/tmp/foo /tmp/bar
```

{% endmethod %}

{% method %}
## Specify the Container

Specify the Container within a Pod running multiple containers.

- `-c <container-name>`

{% sample lang="yaml" %}

```bash
kubectl cp /tmp/foo <some-pod>:/tmp/bar -c <specific-container>
```

{% endmethod %}

{% method %}
## Namespaces

Set the Pod namespace by prefixing the Pod name with `<namespace>/` .

- `<pod-namespace>/<pod-name>:<path>`

{% sample lang="yaml" %}

```bash
kubectl cp /tmp/foo <some-namespace>/<some-pod>:/tmp/bar
```

{% endmethod %}
