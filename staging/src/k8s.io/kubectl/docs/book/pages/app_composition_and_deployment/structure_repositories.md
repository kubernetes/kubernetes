{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/JH35X82)**
{% endpanel %}

{% panel style="warning", title="Experimental" %}
**Content in this chapter is experimental and will evolve based on user feedback.**

Leave feedback on the conventions by creating an issue in the [kubectl](https://github.com/kubernetes/kubectl/issues)
GitHub repository.

Also provide feedback on new kubectl docs at the [survey](https://www.surveymonkey.com/r/JH35X82)
{% endpanel %}


{% panel style="info", title="TL;DR" %}
Decouple changes to Config owned by separate Teams.
{% endpanel %}

# Repository Structure Based Layout

## Motivation

- **Isolation between teams** managing separate Environments
  - Permissions
- **Fine grain control** over
  - PRs
  - Issues
  - Projects
  - Automation

## Directory Structure

### Resource Config

| Repo Type       | Deployed to a Cluster              | Contains | Example Names |
|-----------------|------------------------------------|----------|---------------|
| Base            | **No** - Used as Base              | Config shared with other teams. | `platform` |
| App             | **Yes** - Manually or Continuously | Deployable Config. | `guest-book` |

Use with techniques described in [Directories](structure_directories.md) and [Branches](structure_branches.md)

## Workflow Example

1. Alice on the Java Platform team updates the Java Base used by other teams
1. Alice creates a Git Tag for the new release
1. Bob on the GuestBook App team switches to the new Java Base by updating the reference

## Diagram

### Scenario

1. Alice modifies java Base Repo and tags it v2
  - Change doesn't get pushed anywhere yet
1. Bob modifies GuestBook App Repo to use v2 of the java Base
  - Change gets pushed by continuous deployment

{% sequence width=1000 %}

participant Base Repo as BR
participant App Repo as AR
participant Cluster as C

Note left of BR: Alice: Platform Dev
Note over BR: Alice modifies java Base
Note over BR: Alice tags java Base v2
Note left of AR: Bob: App Dev
Note over AR: Uses java Base v1
BR-->AR: Bob updates to reference Base v2
Note over AR: Uses java Base v2
AR-->C: java Base v2 changes deployed

{% endsequence %}


{% method %}

{% sample lang="yaml" %}

Structure:

- Platform teams create Base Repositories for shared Config
- App teams create App Repositories for their Apps
  - Remotely reference the Base Repository

**Base Repository:** Platform Team

```bash
tree
.
├── bases # Used as a Base only
│   ├── kustomization.yaml
│   └── ...
├── java # Java Bases
│   ├── kustomization.yaml # Uses bases: ["../bases"]
│   └── ...
└── python # Python Bases
```

**App Repository:** GuestBook Team

```bash
tree
.
├── bases # References Base Repositories
│   └── ...
├── prod
│   └── ...
├── staging
│   └── ...
└── test
    └── ...
```

{% endmethod %}

{% panel style="info", title="Remote URLs vs Vendoring" %}
- Repositories owned and controlled by the same organization may be referenced to by their URL
- Repositories owned or controlled by separate organizations should be vendored and referenced
  by path to the vendor directory.
{% endpanel %}

