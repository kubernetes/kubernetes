# Kubernetes CLI/Configuration Roadmap

See also issues with the following labels:
* [area/config-deployment](https://github.com/GoogleCloudPlatform/kubernetes/labels/area%2Fconfig-deployment)
* [component/CLI](https://github.com/GoogleCloudPlatform/kubernetes/labels/component%2FCLI)
* [component/client](https://github.com/GoogleCloudPlatform/kubernetes/labels/component%2Fclient)

1. Create services before other objects, or at least before objects that depend upon them. Namespace-relative DNS mitigates this some, but most users are still using service environment variables. [#1768](https://github.com/GoogleCloudPlatform/kubernetes/issues/1768)
1. Finish rolling update [#1353](https://github.com/GoogleCloudPlatform/kubernetes/issues/1353)
  1. Friendly to auto-scaling [#2863](https://github.com/GoogleCloudPlatform/kubernetes/pull/2863#issuecomment-69701562)
  1. Rollback (make rolling-update reversible, and complete an in-progress rolling update by taking 2 replication controller names rather than always taking a file)
  1. Rollover (replace multiple replication controllers with one, such as to clean up an aborted partial rollout)
  1. Write a ReplicationController generator to derive the new ReplicationController from an old one (e.g., `--image-version=newversion`, which would apply a name suffix, update a label value, and apply an image tag)
  1. Use readiness [#620](https://github.com/GoogleCloudPlatform/kubernetes/issues/620)
  1. Perhaps factor this in a way that it can be shared with [Openshift’s deployment controller](https://github.com/GoogleCloudPlatform/kubernetes/issues/1743) 
  1. Rolling update service as a plugin
1. Kind-based filtering on object streams -- only operate on the kinds of objects specified. This would make directory-based kubectl operations much more useful. Users should be able to instantiate the example applications using `kubectl create -f <example-dir> ...`
1. Improved pretty printing of endpoints, such as in the case that there are more than a few endpoints
1. Service address/port lookup command(s)
1. List supported resources
1. Swagger lookups [#3060](https://github.com/GoogleCloudPlatform/kubernetes/issues/3060)
1. --name, --name-suffix applied during creation and updates
1. --labels and opinionated label injection: --app=foo, --tier={fe,cache,be,db}, --uservice=redis, --env={dev,test,prod}, --stage={canary,final}, --track={hourly,daily,weekly}, --release=0.4.3c2. Exact ones TBD. We could allow arbitrary values -- the keys are important. The actual label keys would be (optionally?) namespaced with kubectl.kubernetes.io/, or perhaps the user’s namespace.
1. --annotations and opinionated annotation injection: --description, --revision
1. Imperative updates. We'll want to optionally make these safe(r) by supporting preconditions based on the current value and resourceVersion.
  1. annotation updates similar to label updates
  1. other custom commands for common imperative updates
  1. more user-friendly (but still generic) on-command-line json for patch
1. We also want to support the following flavors of more general updates:
  1. whichever we don’t support:
    1. safe update: update the full resource, guarded by resourceVersion precondition (and perhaps selected value-based preconditions)
    1. forced update: update the full resource, blowing away the previous Spec without preconditions; delete and re-create if necessary
  1. diff/dryrun: Compare new config with current Spec [#6284](https://github.com/GoogleCloudPlatform/kubernetes/issues/6284)
  1. submit/apply/reconcile/ensure/merge: Merge user-provided fields with current Spec. Keep track of user-provided fields using an annotation -- see [#1702](https://github.com/GoogleCloudPlatform/kubernetes/issues/1702). Delete all objects with deployment-specific labels.
1. --dry-run for all commands
1. Support full label selection syntax, including support for namespaces.
1. Wait on conditions [#1899](https://github.com/GoogleCloudPlatform/kubernetes/issues/1899)
1. Make kubectl scriptable: make output and exit code behavior consistent and useful for wrapping in workflows and piping back into kubectl and/or xargs (e.g., dump full URLs?, distinguish permanent and retry-able failure, identify objects that should be retried)
  1. Here's [an example](http://techoverflow.net/blog/2013/10/22/docker-remove-all-images-and-containers/) where multiple objects on the command line and an option to dump object names only (`-q`) would be useful in combination. [#5906](https://github.com/GoogleCloudPlatform/kubernetes/issues/5906)
1. Easy generation of clean configuration files from existing objects (including containers -- podex) -- remove readonly fields, status
  1. Export from one namespace, import into another is an important use case
1. Derive objects from other objects
  1. pod clone
  1. rc from pod
  1. --labels-from (services from pods or rcs)
1. Kind discovery (i.e., operate on objects of all kinds) [#5278](https://github.com/GoogleCloudPlatform/kubernetes/issues/5278)
1. A fairly general-purpose way to specify fields on the command line during creation and update, not just from a config file
1. Extensible API-based generator framework (i.e. invoke generators via an API/URL rather than building them into kubectl), so that complex client libraries don’t need to be rewritten in multiple languages, and so that the abstractions are available through all interfaces: API, CLI, UI, logs, ... [#5280](https://github.com/GoogleCloudPlatform/kubernetes/issues/5280)
  1. Need schema registry, and some way to invoke generator (e.g., using a container)
  1. Convert run command to API-based generator
1. Transformation framework
  1. More intelligent defaulting of fields (e.g., [#2643](https://github.com/GoogleCloudPlatform/kubernetes/issues/2643))
1. Update preconditions based on the values of arbitrary object fields. 
1. Deployment manager compatibility on GCP: [#3685](https://github.com/GoogleCloudPlatform/kubernetes/issues/3685)
1. Describe multiple objects, multiple kinds of objects [#5905](https://github.com/GoogleCloudPlatform/kubernetes/issues/5905)
1. Support yaml document separator [#5840](https://github.com/GoogleCloudPlatform/kubernetes/issues/5840)

TODO: 
* watch
* attach [#1521](https://github.com/GoogleCloudPlatform/kubernetes/issues/1521)
* image/registry commands
* do any other server paths make sense? validate? generic curl functionality?
* template parameterization
* dynamic/runtime configuration

Server-side support:

1. Default selectors from labels [#1698](https://github.com/GoogleCloudPlatform/kubernetes/issues/1698#issuecomment-71048278)
1. Stop [#1535](https://github.com/GoogleCloudPlatform/kubernetes/issues/1535)
1. Deleted objects [#2789](https://github.com/GoogleCloudPlatform/kubernetes/issues/2789)
1. Clone [#170](https://github.com/GoogleCloudPlatform/kubernetes/issues/170)
1. Resize [#1629](https://github.com/GoogleCloudPlatform/kubernetes/issues/1629)
1. Useful /operations API: wait for finalization/reification
1. List supported resources [#2057](https://github.com/GoogleCloudPlatform/kubernetes/issues/2057)
1. Reverse label lookup [#1348](https://github.com/GoogleCloudPlatform/kubernetes/issues/1348)
1. Field selection [#1362](https://github.com/GoogleCloudPlatform/kubernetes/issues/1362)
1. Field filtering [#1459](https://github.com/GoogleCloudPlatform/kubernetes/issues/1459)
1. Operate on uids


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/cli-roadmap.md?pixel)]()


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/release-0.19.0/docs/cli-roadmap.md?pixel)]()
