# How to update docs for new kubernetes features

Docs github repo: https://github.com/kubernetes/kubernetes.github.io

Instructions for updating the website: http://kubernetes.io/editdocs/ 

**cc *@kubernetes/docs* on your docs update PRs**

## Docs Types To Consider
* Guides
  * Walkthroughs
  * Other Content
* Reference / Glossary
*  Examples

## Content Areas
* API Objects (Pod / Deployment / Service)
* Tools (kubectl / kube-dashboard)
* Cluster Creation + Management

## Questions to ask yourself
* Does this change how any commands are run or the results of running those commands?
  * *Update documentation specifying those commands*
* Should this be present in (or require an update to) one of the walkthroughs?
  * Hellonode
  * K8s101 / k8s201
  * Thorough Walkthrough
* Should this have an overview / dedicated [glossary](http://kubernetes.io/docs/user-guide/images/) section?
  * *Yes for new APIs and kubectl commands*
* Should an existing overview / [glossary](http://kubernetes.io/docs/user-guide/images/) section be updated these changes?
  * *Yes for updates to existing APIs and kubectl commands*
* Should [cluster setup / management](http://kubernetes.io/docs/admin/cluster-management/) guides be updated (which)?  Does this impact all or just some clusters?
* Should [cluster / application debug](https://github.com/kubernetes/kubernetes/wiki/Services-FAQ) guides be updated?
* Should any [tool](http://kubernetes.io/docs/user-guide/kubectl-overview/) guides be updated (kubectl, dashboard)?
* Are there any downstream effects / Does this replace another methodology? (PetSet -> PVC, Deployment -> ReplicationController) - *Which docs for those need to be updated*?
  * Update tutorials to use new style
  * Update examples to use new style
  * Update how tos to use new style
  * Promote new content over old content that it replaces

