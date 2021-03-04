{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/JH35X82)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Continuously Watch and print Resources as they change
{% endpanel %}

# Watching Resources for changes

## Motivation

Print Resources as they are updated.

{% method %}

It is possible to have `kubectl get` **continuously watch for changes to objects**, and print the objects
when they are changed or when the watch is reestablished.

{% sample lang="yaml" %}

```bash
kubectl get deployments --watch
```

```bash
NAME      DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
nginx     1         1         1            1           6h
nginx2    1         1         1            1           21m
```

{% endmethod %}

{% panel style="danger", title="Watch Timeouts" %}
Watch **timesout after 5 minutes**, after which kubectl will re-establish the watch and print the
resources.
{% endpanel %}

{% method %}

It is possible to have `kubectl get` continuously watch for changes to objects **without fetching them first**
using the `--watch-only` flag.

{% sample lang="yaml" %}

```bash
kubectl get deployments --watch-only
```

{% endmethod %}

