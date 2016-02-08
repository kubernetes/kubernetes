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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

Proposed Design
=============

# Goals

Make it really hard to accidentally create a job which has an overlapping selector, while still making it possible to chose an arbitrary selector, and without adding complex constraint solving to the APIserver.

# Use Cases

1. user can leave all label and selector fields blank and system will fill in reasonable ones: non-overlappingness guaranteed.
2. user can put on the pod template some labels that are useful to the user, without reasoning about non-overlappingness.  System adds additional label to assure not overlapping.
3. If user wants to reparent pods to new job (very rare case) and knows what they are doing, they can completely disable this behavior and specify explicit selector.
4. If a controller that makes jobs, like scheduled job, wants to use different labels, such as the time and date of the run, it can do that.
5.  If User reads v1beta1 documentation or reuses v1beta1 Job definitions and just changes the API group, the user should not automatically be allowed to specify a selector, since this is very rarely what people want to do and is error prone.
6. If User downloads an existing job definition, e.g. with `kubectl get jobs/old -o yaml` and tries to modify and post it, he should not create an overlapping job.
7. If User downloads an existing job definition, e.g. with `kubectl get jobs/old -o yaml` and tries to modify and post it, and he accidentally copies the uniquifying label from the old one, then he should not get an error from a label-key conflict, nor get erratic behavior.
8. If user reads swagger docs and sees the selector field, he should not be able to set it without realizing the risks.
8. (Deferred requirement:) If user wants to specify a preferred name for the non-overlappingness key, they can pick a name.

# Proposed changes

## API

`extensions/v1beta1 Job` remains the same. `batch/v1 Job` changes change as follows.

Field `job.spec.noAutoSelector` is added.  It controls whether selectors are automatically
generated.  In automatic mode, user cannot make the mistake of creating non-unique selectors.
In manual mode, certain rare use cases are supported.

Validation is not changed.  A selector must be provided, and it must select the pod template.

Defaulting changes.  Defaulting happens in one of two modes:

### Automatic Mode

- User does not specify `job.spec.selector`.
- User is probably unaware of the `job.spec.noAutoSelector` field and does not think about it.
- User optionally puts labels on pod template (optional).  user does not think about uniqueness, just labeling for user's own reasons.
- Defaulting logic sets `job.spec.selector` to `matchLabels["controller-uid"]="$UIDOFJOB"`
- Defaulting logic  appends 2 labels to the `.spec.template.metadata.labels`.
  - The first label is controller-uid=$UIDOFJOB.
  - The second label is "job-name=$NAMEOFJOB".

### Manual Mode

- User means User or Controller for the rest of this list.
- User does specify `job.spec.selector`.
- User does specify `job.spec.noAutoSelector=true`
- User puts a unique label or label(s) on pod template (required).  user does think carefully about uniqueness.
- No defaulting of pod labels or the selector happen.

### Rationale

UID is better than Name in that:
- it allows cross-namespace control someday if we need it.
- it is unique across all kinds.  `controller-name=foo` does not ensure uniqueness across Kinds `job` vs `replicaSet`.  Even `job-name=foo` has a problem: you might have a `batch.Job` and a `snazzyjob.io/types.Job` -- the latter cannot use label `job-name=foo`, though there is a temptation to do so.
- it uniquely identifies the controller across time.   This prevents the case where, for example, someone deletes a job via the REST api or client (where cascade=false), leaving pods around.  We don't want those to be picked up unintentionally.  It also prevents the case where a user looks at an old job that finished but is not deleted, and tries to select its pods, and gets the wrong impression that it is still running.

Job name is more user friendly.  It is self documenting

Commands like  `kubectl get pods -l job-name=myjob` should do exactly what is wanted 99.9% of the time.  Automated control loops should still use the controller-uid=label.

Using both gets the benefits of both, at the cost of some label verbosity.

The field is a `*bool`.  Since false is expected to be much more common,
and since the feature is complex, it is better to leave it unspecified so that
users looking at a stored pod spec do not need to be aware of this field.

### Overriding Unique Labels

If user does specify `job.spec.selector` then the user must also specify `job.spec.noAutoSelector`.
This ensures the user knows that what he is doing is not the normal thing to do.

To prevent users from copying the `job.spec.noAutoSelector` flag from existing jobs, it will be
optional and default to false, which means when you ask GET and existing job back that didn't use this feature, you don't even see the  `job.spec.noAutoSelector` flag, so you are not tempted to wonder if you should fiddle with it.

## Job Controller

No changes

## Kubectl

No required changes.
Suggest moving SELECTOR to wide output of `kubectl get jobs` since users don't write the selector.

## Docs

Remove examples that use selector and remove labels from pod templates.
Recommend `kubectl get jobs -l job-name=name` as the way to find pods of a job.

# Cross Version Compat

`v1beta1` will not have a `job.spec.noAutoSelector` and will not provide a default selector.

Conversion from v1beta1 to v1 will use the user-provided selector and set `job.spec.noAutoSelector=true`.

# Future Work

Follow this pattern for Deployments, ReplicaSet, DaemonSet when going to v1, if it works well for job.

Docs will be edited to show examples without a `job.spec.selector`.

We probably want as much as possible the same behavior for Job and ReplicationController.






<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/selector-generation.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
