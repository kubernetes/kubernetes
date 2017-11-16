# Example Guidelines

## An Example Is

An example demonstrates running an application/framework/workload on
Kubernetes in a meaningful way. It is educational and informative.

Examples are not:

* Full app deployments, ready to use, with no explanation. These
  belong to [Helm charts](https://github.com/kubernetes/charts).
* Simple toys to show how to use a Kubernetes feature. These belong in
  the [user guide](https://kubernetes.io/docs/user-guide/).
* Demos that follow a script to show a Kubernetes feature in
  action. Example: killing a node to demonstrate controller
  self-healing.
* A tutorial which guides the user through multiple progressively more
  complex deployments to arrive at the final solution. An example
  should just demonstrate how to setup the correct deployment

## An Example Includes

### Up front

* Has a "this is what you'll learn" section.
* Has a Table of Contents.
* Has a section that brings up the app in the fewest number of
  commands (TL;DR / quickstart), without cloning the repo (kubectl
  apply -f http://...).
* Points to documentation of prerequisites.
  * [Create a cluster](https://kubernetes.io/docs/getting-started-guides/) (e.g., single-node docker).
  * [Setup kubectl](https://kubernetes.io/docs/user-guide/prereqs.md).
  * etc.
* Should specify which release of Kubernetes is required and any other
  prerequisites, such as DNS, a cloudprovider with PV provisioning, a
  cloudprovider with external load balancers, etc.
  * Point to general documentation about alternatives for those
    mechanisms rather than present the alternatives in each example.
  * Tries to balance between using using new features, and being
    compatible across environments.

### Throughout

* Should point to documentation on first mention:
  [kubectl](https://kubernetes.io/docs/user-guide/kubectl-overview.md),
  [pods](https://kubernetes.io/docs/user-guide/pods.md),
  [services](https://kubernetes.io/docs/user-guide/services.md),
  [deployments](https://kubernetes.io/docs/user-guide/deployments.md),
  [replication controllers](https://kubernetes.io/docs/user-guide/replication-controller.md),
  [jobs](https://kubernetes.io/docs/user-guide/jobs.md),
  [labels](https://kubernetes.io/docs/user-guide/labels.md),
  [persistent volumes](https://kubernetes.io/docs/user-guide/persistent-volumes.md),
  etc.
* Most examples should be cloudprovider-independent (e.g., using PVCs, not PDs).
  * Other examples with cloudprovider-specific bits could be somewhere else.
* Actually show the app working -- console output, and or screenshots.
  * Ascii animations and screencasts are recommended.
* Follows [config best practices](https://kubernetes.io/docs/user-guide/config-best-practices.md).
* Shouldn't duplicate the [thorough walk-through](https://kubernetes.io/docs/user-guide/#thorough-walkthrough).
* Docker images are pre-built, and source is contained in a subfolder.
  * Source is the Dockerfile and any custom files needed beyond the
    upstream app being packaged.
  * Images are pushed to `gcr.io/google-samples`. Contact @jeffmendoza
    to have an image pushed
  * Images are tagged with a version (not latest) that is referenced
    in the example config.
* Only use the code highlighting types
  [supported by Rouge](https://github.com/jneen/rouge/wiki/list-of-supported-languages-and-lexers),
  as this is what GitHub Pages uses.
* Commands to be copied use the `shell` syntax highlighting type, and
  do not include any kind of prompt.
* Example output is in a separate block quote to distinguish it from
  the command (which doesn't have a prompt).
* When providing an example command or config for which the user is
  expected to substitute text with something specific to them, use
  angle brackets: `<IDENTIFIER>` for the text to be substituted.
* Use `kubectl` instead of `cluster\kubectl.sh` for example cli
  commands.

### At the end

* Should have a section suggesting what to look at next, both in terms
  of "additional resources" and "what example to look at next".


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/guidelines.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
