{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/JH35X82)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Execute a Command in a Container
- Get a Shell in a Container
{% endpanel %}

# Executing Commands

## Motivation

Debugging Workloads by running commands within the Container.  Commands may be a Shell with
a tty.

{% method %}
## Exec Command

Run a command in a Container in the cluster by specifying the **Pod name**.

{% sample lang="yaml" %}

```bash
kubectl exec nginx-78f5d695bd-czm8z ls
```

```bash
bin  boot  dev	etc  home  lib	lib64  media  mnt  opt	proc  root  run  sbin  srv  sys  tmp  usr  var
```

{% endmethod %}

{% method %}
## Exec Shell

To get a Shell in a Container, use the `-t -i` options to get a tty and attach STDIN.

{% sample lang="yaml" %}

```bash
kubectl exec -t -i nginx-78f5d695bd-czm8z bash
```

```bash
root@nginx-78f5d695bd-czm8z:/# ls
bin  boot  dev	etc  home  lib	lib64  media  mnt  opt	proc  root  run  sbin  srv  sys  tmp  usr  var
```

{% endmethod %}

{% panel style="info", title="Specifying the Container" %}
For Pods running multiple Containers, the Container should be specified with `-c <container-name>`.
{% endpanel %}
