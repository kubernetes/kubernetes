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
Decouple changes to Config to be deployed to separate Environments.
{% endpanel %}

# Branch Structure Based Layout

## Motivation

**Why use branches?** Decouple changes that are rolled out with releases (e.g. new flags) from changes that are
rolled out in response to production events (e.g. resource tuning).

## Branch Structure

The convention shown here should be changed and adapted as needed.

| Branch Type Name                                   | Deployed to a Cluster | Purpose  |     Example Config Change        | Example Branch Name |
|----------------------------------------|----|-----------|--------|----|
| Base   | **No**.  Merged into other Branches only. | Changes that should be rolled out as part of a release. | Add *pubsub topic* flag | `master`, `release-1.14`, `i1026` |
| Deploy   | **Yes**. - Manually or Continuously.  | Base + Changes required to respond to "production" events (or dev, staging, etc). | Increase *memory resources* - e.g. for crashing Containers | `deploy-test`, `deploy-staging`, `deploy-prod` |

Use with techniques described in [Directories](structure_directories.md) and [Branches](structure_branches.md)

## Workflow Example

### Diagram

#### Scenario

1. Live Prod App version is *v1*
1. *v2* changes committed to Base Branch Config
1. *v2* rolled out to Staging
  - Deployed by continuous deployment
1. Live Prod App requires change to *v1* (unrelated to *v2*)
  - Change memory resources in Prod
1. Prod Branch Config Updated at *v1*
  - Deployed immediately by continuous deployment
1. *v2* changes rolled out separately
 - Tag on Base Branch merged into Prod Branch
 - Prod Branch continuously deployed

{% sequence width=1000 %}

participant Base Branch as BB
participant Staging Branch as SB
participant Staging Clusters as SC
participant Prod Branch as PB
participant Prod Clusters as PC

Note over SC: At v1 release
Note over PC: At v1 release
Note left of BB: Bob: App Dev
Note over BB: Bob Adds Flag
Note over BB: Bob Tags v2
Note over SB: Bob Releases v2
BB-->SB: Merge v2
SB-->SC: Deploy
Note over SC: At v2 release
Note over BB,PC: Prod Outage
Note left of PB: Alice: App SRE
Note over PB: Alice fixes Config
PB-->PC: Alice's changes (only)
Note over PC: At v1* release
Note over BB,PC: Prod Outage resolved
Note over PB: Alice Releases v2
BB-->PB: Merge v2
PB-->PC: Deploy v2
Note over PC: At v2 release

{% endsequence %}

### Description

**Note:** Starting version of Application is *v1*

1. Developer Bob introduces new app flag for release with *v2*
  - e.g. PubSub topic name
1. Bob updates the Base Config with the new flag
  - Add staging topic for Staging (e.g. `staging-topic`)
  - Add prod topic for Prod (e.g. `prod-topic`)
  - Flag should be rolled out with *v2* release
1. *v2* is cut
  - Base tagged with *v2* tag
1. *v2* rolled out to Staging
  - Merge *v2* Tag -> Staging Branch
  - Deploy Staging Branch to Staging Clusters
1. SRE Alice identifies issue in Prod (at *v1*)
  - Fix is to increase memory of containers
1. Alice updates the Prod branch Config by increasing memory resources
  - Changes go directly into Prod Branch without going into Base
1. *v1* changes rolled out to Prod (*v1++*)
  - Include Alice's changes, but not Bob's
1. *v2* rolled out to Prod
  - Merge *v2* Tag -> Prod Branch
  - Deploy Prod Branch to Prod Clusters

{% method %}

Techniques:

- Add new required flags and environment variables to the Resource Config in the Base branch at the
  time they are added to the code.
  - Will be rolled out when the code is rolled out.
- Adjust flags and configuration to the Resource Config in the Deploy branch in the deploy directory.
  - Will be rolled out immediately independent of versions.
- Merge code from the Base branch to the Deploy branches to perform a Rollout.

## Directory and Branch Layout

Structure:

- Base branch (e.g. `master`, `app-version`, etc) for Config changes tied to releases.
  - Looks like [Directories](structure_directories.md)
- Separate Deploy branches for separate Environments (e.g. `deploy-<env>`).
  - A new **Directory in each branch with will contain overlay customizations** - e.g. `deploy-<env>`.

{% sample lang="yaml" %}

**Base Branch:** `master`

```bash
tree
.
├── bases
│   ├── ...
├── prod
│   ├── bases
│   │   ├── ...
│   ├── us-central
│   │   ├── kustomization.yaml
│   │   └── backend
│   │       └── deployment-patch.yaml
│   ├── us-east
│   │   └── kustomization.yaml
│   └── us-west
│       └── kustomization.yaml
├── staging
│   ├── bases
│   │   ├── ...
│   └── us-west
│       └── kustomization.yaml
└── test
    ├── bases
    │   ├── ...
    └── us-west
        └── kustomization.yaml
```

**Deploy Branches:**

Prod Branch: `deploy-prod`

```bash
tree
.
├── bases # From Base Branch
│   └── ...
└── deploy-prod # Prod deploy folder
│   ├── us-central
│   │   ├── kustomization.yaml # Uses bases: ["../../prod/us-central"]
│   ├── us-east
│   │   └── kustomization.yaml # Uses bases: ["../../prod/us-east"]
│   └── us-west
│       └── kustomization.yaml # Uses bases: ["../../prod/us-west"]
├── prod # From Base Branch
│   └── ...
├── staging # From Base Branch
│   └── ...
└── test # From Base Branch
    └── ...
```

Staging Branch: `deploy-staging`

```bash
tree
.
├── bases # From Base Branch
│   ├── ...
├── deploy-staging # Staging deploy folder
│   └── us-west
│       └── kustomization.yaml # Uses bases: ["../../staging/us-west"]
├── prod # From Base Branch
│   └── ...
├── staging # From Base Branch
│   └── ...
└── test # From Base Branch
    └── ...
```

Test Branch: `deploy-test`

```bash
tree
.
├── bases # From Base Branch
│   ├── ...
├──deploy-test # Test deploy folder
│   └── us-west
│       └── kustomization.yaml # Uses bases: ["../../test/us-west"]
├── prod # From Base Branch
│   └── ...
├── staging # From Base Branch
│   └── ...
└── test # From Base Branch
    └── ...
```

{% endmethod %}

## Rollback Workflow Example

Summary of rollback workflow with Branches:

1. Live Prod App version is *v1*
1. Changes are introduced to Base Branch Config
  - To be released with version *v2*
1. Release *v2* is cut to be rolled out
  - Tag Base *v2* and build artifacts (e.g. images)
1. Changes are introduced into the Base Branch Confiug
  - To be released with version *v3*
1. *v2* is pushed to Prod (eventually)
  - *v2* Tag merged into Prod Branch
1. *v2* has issues in Prod and must be rolled back
  - *v2* changes are rolled back in new commit to Prod Branch
1. Base Branch is unaffected
  - Fix introduced in *v3*

**Note:** New changes committed to the Base for "v3" did not make the rollback from
"v2" -> "v1" more challenging, as they had not been merged into the Prod Branch.

### Diagram

{% sequence width=1000 %}

participant Base Branch as BB
participant Staging Branch as SB
participant Staging Clusters as SC
participant Prod Branch as PB
participant Prod Clusters as PC

Note over SC: At v1 release
Note over PC: At v1 release
Note left of BB: Bob: App Dev
Note over BB: Bob Adds Flag (for v2)
Note over BB: Bob Tags v2
Note over SB: Bob Releases v2
BB-->SB: Merge v2
SB-->SC: Deploy
Note over SC: At v2 release
Note over SB: Bob Adds another Flag (for v3)
Note over PB: Bob Releases v2
BB-->PB: Merge v2
PB-->PC: Deploy v2
Note over PC: At v2 release
Note over BB,PC: Unrelated Prod Outage
Note left of PB: Alice: App SRE
Note over PB: Alice rolls back v2 merge commit
PB-->PC: Deploy v1
Note over PC: At v1 release
Note over BB,PC: Prod Outage resolved

{% endsequence %}
