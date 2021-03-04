{% panel style="success", title="Providing Feedback" %}
**Provide feedback at the [survey](https://www.surveymonkey.com/r/JH35X82)**
{% endpanel %}

{% panel style="info", title="TL;DR" %}
- Resource Config is stored in one or more git repositories
- Directory hierarchy, git branches and git repositories may be used for loose coupling
{% endpanel %}


# Resource Config Structure

The chapters in this section cover how to structure Resource Config using git.

Users may start with a pure Directory Hierarchy approach, and later include Branches
and / or Repositories as part of the structure.

## Background

Terms:

- *Bases:* provide **common or shared Resource Config to be factored out** that can be
  imported into multiple projects.
- *Overlays and Customizations:* tailor **common or shared Resource Config to be modified** to
  a specific application, environment or purpose.

| Technique                                   | Decouple Changes            | Used For                                           | Workflow |
|---------------------------------------------|-----------------------------|----------------------------------------------------|----------|
| [Directories](structure_directories.md)     | NA                          | Foundational structure.    | Changes are immediately propagated globally.  |
| [Branches](structure_branches.md)           | *Across Environments*       | Promoting changes across Environments. | Changes are promoted across linear stages. |
| [Repositories](structure_repositories.md)   | *Across Teams*              | Fetching changes across config shared across Teams. | Changes are pulled by consumers (like upgrades). |

Concepts:

- Resource Config may be initially structured using only Directory Hierarchy for organization.
  - Use Bases with Overlays / Customizations for factoring across Directories
- Different Deployment environments for the same app may be loosely coupled
  - Use separate **Branches for separate environments**.
  - Use Bases with Overlays / Customization for factoring across Branches
- Different Teams owning sharing Config may be loosely coupled
  - Use separate **Repositories for separate teams**.
  - Use Bases with Overlays / Customization for factoring across Repositories

