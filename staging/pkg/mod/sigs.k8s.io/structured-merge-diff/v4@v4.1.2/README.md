# Structured Merge and Diff

This repo contains code which implements the Kubernetes "apply" operation.

## What is the apply operation?

We model resources in a control plane as having multiple "managers". Each
manager is typically trying to manage only one aspect of a resource. The goal is
to make it easy for disparate managers to make the changes they need without
messing up the things that other managers are doing. In this system, both humans
and machines (aka "controllers") act as managers.

To do this, we explicitly track (using the fieldset data structure) which fields
each manager is currently managing.

Now, there are two basic mechanisms by which one modifies an object.

PUT/PATCH: This is a write command that says: "Make the object look EXACTLY like
X".

APPLY: This is a write command that says: "The fields I manage should now look
exactly like this (but I don't care about other fields)".

For PUT/PATCH, we deduce which fields will be managed based on what is changing.
For APPLY, the user is explicitly stating which fields they wish to manage (and
therefore requesting deletion of any fields that they used to manage but stop
mentioning).

Any time a manager begins managing some new field, that field is removed from
all other managers. If the manager is using the APPLY command, we call these
conflicts, and will not proceed unless the user passes the "force" option. This
prevents accidentally setting fields which some other entity is managing.

PUT/PATCH always "force". They are mostly used by automated systems, which won't
do anything productive with a new error type.

## Components

The operation has a few building blocks:

* We define a targeted schema type in the schema package. (As a consequence of
  being well-targeted, it's much simpler than e.g. OpenAPI.)
* We define a "field set" data structure, in the fieldpath package. A field path
  locates a field in an object, generally a "leaf" field for our purposes. A
  field set is a group of such paths.  They can be stored efficiently in what
  amounts to a Trie.
* We define a "value" type which stores an arbitrary object.
* We define a "typed" package which combines "value" and "schema". Now we can
  validate that an object conforms to a schema, or compare two objects.
* We define a "merge" package which uses all of the above concepts to implement
  the "apply" operation.
* We will extensively test this.

## Community, discussion, contribution, and support

Learn how to engage with the Kubernetes community on the [community page](http://kubernetes.io/community/).

You can reach the maintainers of this project at:

- Slack: [#wg-api-expression](https://kubernetes.slack.com/messages/wg-api-expression)
- [Mailing List](https://groups.google.com/forum/#!forum/kubernetes-wg-api-expression)

### Code of conduct

Participation in the Kubernetes community is governed by the [Kubernetes Code of Conduct](code-of-conduct.md).

[owners]: https://git.k8s.io/community/contributors/guide/owners.md
