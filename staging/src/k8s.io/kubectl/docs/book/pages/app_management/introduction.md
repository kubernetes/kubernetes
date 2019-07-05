{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/CLQBQHR)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Apply manages Applications through files defining Kubernetes Resources (i.e. Resource Config)
- Kustomize is used to author Resource Config
{% endpanel %}


# Declarative Application Management

This section covers how to declaratively manage Workloads and Applications.

Workloads in a cluster may be configured through files called *Resource Config*.  These files are
typically checked into source control, and allow cluster state changes to be reviewed before they
are audited and applied.

There are 2 components to Application Management.

## Client Component

The client component consists of authoring Resource Config which defines the desired state
of an Application. This may be done as a collection of raw Resource Config files, or by
composing and overlaying Resource Config authored by separate teams
(using the `-k` flag with a `kustomization.yaml`).

Kustomize offers low-level tooling for simplifying the authoring of Resource Config.  It provides:

- **Generating Resource Config** from other canonical sources - e.g. ConfigMaps, Secrets
- **Reusing and Composing one or more collections of Resource Config**
- **Customizing Resource Config**
- **Setting cross-cutting fields** - e.g. namespace, labels, annotations, name-prefixes, etc

**Example:** One user may define a Base for an application,  while another user may customize
a specific instance of the Base.

## Server Component

The server component consists of a human applying the authored Resource Config to the cluster
to create or update Resources.  Once Applied, the Kubernetes cluster will set additional desired
state on the Resource - e.g. *defaulting unspecified fields, filling in IP addresses, autoscaling
replica count, etc.*

Note that the process of Application Management is a collaborative one between users and the
Kubernetes system itself - where each may contribute to defining the desired state.

**Example**: An Autoscaler Controller in the cluster may set the scale field on a Deployment managed by a user.
