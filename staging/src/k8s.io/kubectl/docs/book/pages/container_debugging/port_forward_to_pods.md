{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/JH35X82)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Port Forward local connections to Pods running in a cluster 
{% endpanel %}

# Port Forward

## Motivation

Connect to ports of Pods running a cluster by port forwarding local ports.

{% method %}
## Forward Multiple Ports

Listen on ports 5000 and 6000 locally, forwarding data to/from ports 5000 and 6000 in the pod
{% sample lang="yaml" %}

```bash
kubectl port-forward pod/mypod 5000 6000
```

{% endmethod %}

---

{% method %}
## Pod in a Workload

Listen on ports 5000 and 6000 locally, forwarding data to/from ports 5000 and 6000 in a pod selected by the
deployment
{% sample lang="yaml" %}

```bash
kubectl port-forward deployment/mydeployment 5000 6000
```

{% endmethod %}

---

{% method %}
## Pod in a Service

Listen on port 8443 locally, forwarding to the targetPort of the service's port named "https" in a pod selected by the service
{% sample lang="yaml" %}

```bash
kubectl port-forward service/myservice 8443:https
```

{% endmethod %}

---

{% method %}
## Different Local and Remote Ports

Listen on port 8888 locally, forwarding to 5000 in the pod
{% sample lang="yaml" %}

```bash
kubectl port-forward pod/mypod 8888:5000
```

{% endmethod %}

---

{% method %}
## Random Local Port

Listen on a random port locally, forwarding to 5000 in the pod
{% sample lang="yaml" %}

```bash
kubectl port-forward pod/mypod :5000
```

{% endmethod %}
