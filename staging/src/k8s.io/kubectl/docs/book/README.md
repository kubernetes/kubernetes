{% panel style="success", title="Feedback and Contributing" %}
**Provide feedback on new kubectl docs at the [survey](https://www.surveymonkey.com/r/JH35X82)**

See [CONTRIBUTING](https://github.com/kubernetes/kubectl/blob/master/docs/book/CONTRIBUTING.md) for
instructions on filing/fixing issues and adding new content.
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Kubectl is the Kubernetes cli
- Kubectl provides a swiss army knife of functionality for working with Kubernetes clusters
- Kubectl may be used to deploy and manage applications on Kubernetes
- Kubectl may be used for scripting and building higher-level frameworks
{% endpanel %}

# Kubectl

Kubectl is the Kubernetes cli version of a swiss army knife, and can do many things.

While this Book is focused on using Kubectl to declaratively manage Applications in Kubernetes, it
also covers other Kubectl functions.

## Command Families

Most Kubectl commands typically fall into one of a few categories:

| Type                                   | Used For                   | Description                                        |
|----------------------------------------|----------------------------|----------------------------------------------------|
| Declarative Resource Management        | Deployment and Operations (e.g. GitOps)   | Declaratively manage Kubernetes Workloads using Resource Config     |
| Imperative Resource Management         | Development Only           | Run commands to manage Kubernetes Workloads using Command Line arguments and flags |
| Printing Workload State | Debugging  | Print information about Workloads |
| Interacting with Containers | Debugging  | Exec, Attach, Cp, Logs |
| Cluster Management | Cluster Ops | Drain and Cordon Nodes |

## Declarative Application Management

The preferred approach for managing Resources is through
declarative files called Resource Config used with the Kubectl *Apply* command.
This command reads a local (or remote) file structure and modifies cluster state to
reflect the declared intent.

{% panel style="info", title="Apply" %}
Apply is the preferred mechanism for managing Resources in a Kubernetes cluster.
{% endpanel %}

## Printing state about Workloads

Users will need to view Workload state.

- Printing summarize state and information about Resources
- Printing complete state and information about Resources
- Printing specific fields from Resources
- Query Resources matching labels

## Debugging Workloads

Kubectl supports debugging by providing commands for:

- Printing Container logs
- Printing cluster events
- Exec or attaching to a Container
- Copying files from Containers in the cluster to a user's filesystem

## Cluster Management

On occasion, users may need to perform operations to the Nodes of cluster.  Kubectl supports
commands to drain Workloads from a Node so that it can be decommission or debugged.

## Porcelain

Users may find using Resource Config overly verbose for *Development* and prefer to work with
the cluster *imperatively* with a shell-like workflow.  Kubectl offers porcelain commands for
generating and modifying Resources.

- Generating + creating Resources such as Deployments, StatefulSets, Services, ConfigMaps, etc
- Setting fields on Resources
- Editing (live) Resources in a text editor

{% panel style="danger", title="Porcelain For Dev Only" %}
Porcelain commands are time saving for experimenting with workloads in a dev cluster, but shouldn't
be used for production.
{% endpanel %}
