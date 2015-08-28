<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/proposals/node-attribute-labels.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Export Node Attributes as Labels

## Motivation

Kubernetes assumes a homogeneous cluster of nodes and does not facilitate scheduling
to a subset defined by one or more arbitrary attributes.  Labels may be applied
manually to nodes through the api, but cannot be defined as part of the node configuration.
One such use case for this would be hardening a set of nodes for security and restrict
certain pods to these nodes for regulatory reasons.

## Goals

Allow Kubernetes administrators and cloud providers to easily configure arbitrary
node attribute labels.  These labels would be available for specifying pod
scheduling requirements.  Examples could include `docker.node.kubernetes.io/version=docker://1.5`,
`aws.node.kubernetes.io/az=us-east-1`, or `k.node.kubernetes.io/version=1.1`

## Non-Goals

Supporting labels for consumable / finite attributes.  "Attributes"
such as SSD and GPU are consumable and should be represented as generic resources.
See [generic resources](../../docs/design/resources.md).

## Proposal

### Label Namespace

Node attribute labels will live under the namespace `<some prefix>.node.kubernetes.io` and be required
to match the glob `*.node.kubernetes.io/*`.  The name space `node.kubernetes.io` is reserved for use by
kubernetes.

### Plugins

We will allow cloud providers and kubernetes administrators to specify custom node attribute
labels through plugins.  Plugins will be checked at node start up time and published
to the api server as part of the Node.ObjectMeta.Labels structure.  The plugin directory
can be specified via the kubelet server command line argument `node-label-plugin-dir`.
This flag specifies a directory containing one or more executables or json files.  This directory
should NOT be mountable by containers via HostPath.  At kubelet initialization time these files
will be used to apply additional labels to the node.  After start up, plugins will be polled periodically
(every 10 minutes) in a background thread to update the labels sent as part of Node.ObjectMeta.Labels
during health checks.  If there are errors while updating the labels, then no updates will be applied
and the last good set will be used.  If there is an error during the first time plugins are checked,
then no additional labels will be applied but the non-plugin specified node labels will still be used.

## Plugin File Format

### Json

Json file plugins will be identified by having the extension '.json' and expected to be a map[string]string.
e.g. a file named `label.json` with contents`{"a.node.kubernetes.io/l1":"v1", "a.node.kubernetes.io/l2":"v2"}`

## Executable

Executable file plugins will be identified by having a white listed extension.  The initial whitelist will
be ['.sh', '.py'].  Executable file plugins are expected to be executable by the kubelet.  If the
kubelet cannot execute a plugin identified by its extension, it is considered an error and no labels
will be updated.  Executable plugins are expected to emit to stdout json that can be parsed as a json plugin.
e.g. a file name `label.sh` with contents `echo '{"a.node.kubernetes.io/l1":"v1", "a.node.kubernetes.io/l2":"v2"}'`

### Export of Labels to Api Server

Labels will be exported to the Api server as part of the api.Node.ObjectMeta.Labels map at registration
and during health checks.

## Other Considerations

- Node labels are not versioned and cannot easily be changed.  We should be careful about the labels we apply.
- Label updates are done via periodic polling.  There are more efficient ways of getting updates at the cost of
additional complexity (e.g. how do we get them from executable plugins?).

## Open questions

- __Do we really need to support executables as a plugin format?  Is supporting json files only sufficient?__
  - A lot of the complexity is around handling executables.  It may be better to hold off on supporting exec.
- Are there security concerns with directly executing plugin files based on their location?
- Are there bandwidth concerns about sending the full set of labels on each health check?
- Should we be worried about frequently running plugins that are expensive to run?
- Should we put restrictions on the amount of resources plugins may consume?
- Should we support arbitrary executable plugins based on files permissions, or just a whitelist suffixes (.sh, .py)?
- Do we need to export monitoring metrics so failures to run plugins?


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/node-attribute-labels.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
