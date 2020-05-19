{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/JH35X82)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Format and print specific fields from Resources
- Use when scripting with Get
{% endpanel %}

# Print Resource Fields

## Motivation

Kubectl Get is able to pull out fields from Resources it queries and format them as output.

This may be **useful for scripting or gathering data** about Resources from a Kubernetes cluster.

## Get

The `kubectl get` reads Resources from the cluster and formats them as output.  The examples in
this chapter will query for Resources by providing Get the *Resource Type* with the
Version and Group as an argument.
For more query options see [Queries and Options](queries_and_options.md).

Kubectl can format and print specific fields from Resources using Json Path.

{% panel style="warning", title="Scripting Pitfalls" %}
By default, if no API group or version is specified, kubectl will use the group and version preferred by
the apiserver.

Because the **Resource structure may change between API groups and Versions**, users *should* specify the
API Group and Version when emitting fields from `kubectl get` to make sure the command does not break
in future releases.

Failure to do this may result in the different API group / version being used after a cluster upgrade, and
this group / version may have changed the representation of fields.
{% endpanel %}

### JSON Path

Print the fields from the JSON Path

**Note:**  JSON Path can also be read from a file using `-o custom-columns-file`.

- JSON Path template is composed of JSONPath expressions enclosed by {}. In addition to the original JSONPath syntax, several capabilities are added:
  - The `$` operator is optional (the expression starts from the root object by default).
  - Use "" to quote text inside JSONPath expressions.
  - Use range operator to iterate lists.
  - Use negative slice indices to step backwards through a list. Negative indices do not “wrap around” a list. They are valid as long as -index + listLength >= 0.

### JSON Path Symbols Table

| Function	| Description	| Example	| Result |
|---|---|---|---|
| text	| the plain text	| kind is {.kind}	| kind is List |
| @	| the current object	| {@}	| the same as input |
| . or [] |	child operator	| {.kind} or {[‘kind’]}	| List |
| ..	| recursive descent	| {..name}	| 127.0.0.1 127.0.0.2 myself e2e |
| *	| wildcard. Get all objects	| {.items[*].metadata.name}	| [127.0.0.1 127.0.0.2] |
| [start:end :step]	| subscript operator	| {.users[0].name}	| myself |
| [,]	| union operator	| {.items[*][‘metadata.name’, ‘status.capacity’]}	|127.0.0.1 127.0.0.2 map[cpu:4] map[cpu:8] |
| ?()	| filter	| {.users[?(@.name==“e2e”)].user.password}	| secret |
| range, end	| iterate list	| {range .items[*]}[{.metadata.name}, {.status.capacity}] {end}	| [127.0.0.1, map[cpu:4]] [127.0.0.2, map[cpu:8]] |
| “	| quote interpreted string	| {range .items[*]}{.metadata.name}{’\t’} {end} |	127.0.0.1 127.0.0.2|

---

{% method %}

Print the JSON representation of the first Deployment in the list on a single line.

{% sample lang="yaml" %}

```bash
kubectl get deployment.v1.apps -o=jsonpath='{.items[0]}{"\n"}'

```

```bash
map[apiVersion:apps/v1 kind:Deployment...replicas:1 updatedReplicas:1]]
```
{% endmethod %}

---

{% method %}

Print the `metadata.name` field for the first Deployment in the list.

{% sample lang="yaml" %}

```bash
kubectl get deployment.v1.apps -o=jsonpath='{.items[0].metadata.name}{"\n"}'
```

```bash
nginx
```

{% endmethod %}

---

{% method %}

For each Deployment, print its `metadata.name` field and a newline afterward.

{% sample lang="yaml" %}

```bash
kubectl get deployment.v1.apps -o=jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'
```

```bash
nginx
nginx2
```

{% endmethod %}

---

{% method %}

For each Deployment, print its `metadata.name` and `.status.availableReplicas`.

{% sample lang="yaml" %}

```bash
kubectl get deployment.v1.apps -o=jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.availableReplicas}{"\n"}{end}'
```
```bash
nginx	1
nginx2	1
```

{% endmethod %}

---

{% method %}

Print the list of Deployments as single line.

{% sample lang="yaml" %}

```bash
kubectl get deployment.v1.apps -o=jsonpath='{@}{"\n"}'
```

```bash
map[kind:List apiVersion:v1 metadata:map[selfLink: resourceVersion:] items:[map[apiVersion:apps/v1 kind:Deployment...replicas:1 updatedReplicas:1]]]]
```

{% endmethod %}

---

{% method %}

Print each Deployment on a new line.

{% sample lang="yaml" %}

```bash
kubectl get deployment.v1.apps -o=jsonpath='{range .items[*]}{@}{"\n"}{end}'
```

```bash
map[kind:Deployment...readyReplicas:1]]
map[kind:Deployment...readyReplicas:1]]
```

{% endmethod %}

---

{% panel style="info", title="Literal Syntax" %}
On Windows, you must double quote any JSONPath template that contains spaces (not single quote as shown above for bash).
This in turn means that you must use a single quote or escaped double quote around any literals in the template.

For example:

```bash
C:\> kubectl get pods -o=jsonpath="{range .items[*]}{.metadata.name}{'\t'}{.status.startTime}{'\n'}{end}"
```
{% endpanel %}
