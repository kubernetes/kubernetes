{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/JH35X82)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- View diff of changes before they are Applied to the cluster
{% endpanel %}

# Diffing Local and Cluster State

## Motivation

The ability to view what changes will be made before applying them to a cluster can be useful.

{% method %}
## Generating a Diff

Use the `diff` program in a user's path to display a diff of the changes that will be
made by Apply.

{% sample lang="yaml" %}

```sh
kubectl diff -k ./dir/
```

{% endmethod %}

{% method %}
## Setting the Diff Program

The `KUBECTL_EXTERNAL_DIFF` environment variable can be used to select your own diff command.
By default, the "diff" command available in your path will be run with "-u" (unified) and "-N"
(treat new files as empty) options.


{% sample lang="yaml" %}

```sh
export KUBECTL_EXTERNAL_DIFF=meld; kubectl diff -k ./dir/
```

{% endmethod %}

