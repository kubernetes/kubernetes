{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/JH35X82)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Print the Logs of a Container in a cluster
{% endpanel %}

# Summarizing Resources

## Motivation

Debugging Workloads by printing out the Logs of containers in a cluster.

{% method %}
## Print Logs for a Container in a Pod

Print the logs for a Pod running a single Container
{% sample lang="yaml" %}

```bash
kubectl logs echo-c6bc8ccff-nnj52
```

```bash
hello
hello
```

{% endmethod %}


{% panel style="success", title="Crash Looping Containers" %}
If a container is crash looping and you want to print its logs after it
exits, use the `-p` flag to look at the **logs from containers that have
exited**.  e.g. `kubectl logs -p -c ruby web-1`
{% endpanel %}

---

{% method %}
## Print Logs for all Pods for a Workload

Print the logs for all Pods for a Workload
{% sample lang="yaml" %}

```bash
# Print logs from all containers matching label
kubectl logs -l app=nginx
```

{% endmethod %}

{% panel style="success", title="Workloads Logs" %}
Print all logs from **all containers for a Workload** by passing the
Workload label selector to the `-l` flag.  e.g. if your Workload
label selector is `app=nginx` usie `-l "app=nginx"` to print logs
for all the Pods from that Workload.
{% endpanel %}

---

{% method %}
## Follow Logs for a Container

Stream logs from a container.

{% sample lang="yaml" %}

```bash
# Follow logs from container
kubectl logs nginx-78f5d695bd-czm8z -f
```

{% endmethod %}

---

{% method %}
## Printing Logs for a Container that has exited

Print the logs for the previously running container.  This is useful for printing containers that have
crashed or are crash looping.
{% sample lang="yaml" %}

```bash
# Print logs from exited container
kubectl logs nginx-78f5d695bd-czm8z -p
```

{% endmethod %}

---

{% method %}
## Selecting a Container in a Pod 

Print the logs from a specific container within a Pod.  This is necessary for Pods running multiple
containers.
{% sample lang="yaml" %}

```bash
# Print logs from the nginx container in the nginx-78f5d695bd-czm8z Pod
kubectl logs nginx-78f5d695bd-czm8z -c nginx
```

{% endmethod %}

---

{% method %}
## Printing Logs After a Time

Print the logs that occurred after an absolute time.
{% sample lang="yaml" %}

```bash
# Print logs since a date
kubectl logs nginx-78f5d695bd-czm8z --since-time=2018-11-01T15:00:00Z
```

{% endmethod %}

---

{% method %}
## Printing Logs Since a Time

Print the logs that are newer than a duration.

Examples:

- 0s: 0 seconds
- 1m: 1 minute
- 2h: 2 hours

{% sample lang="yaml" %}

```bash
# Print logs for the past hour
kubectl logs nginx-78f5d695bd-czm8z --since=1h
```

{% endmethod %}

---

{% method %}
## Include Timestamps

Include timestamps in the log lines

{% sample lang="yaml" %}

```bash
# Print logs with timestamps
kubectl logs -l app=echo --timestamps
```

```bash
2018-11-16T05:26:31.38898405Z hello
2018-11-16T05:27:13.363932497Z hello
```

{% endmethod %}