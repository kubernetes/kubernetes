<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Annotations

We have [labels](labels.md) for identifying metadata.

It is also useful to be able to attach arbitrary non-identifying metadata, for retrieval by API clients such as tools, libraries, etc. This information may be large, may be structured or unstructured, may include characters not permitted by labels, etc. Such information would not be used for object selection and therefore doesn't belong in labels.

Like labels, annotations are key-value maps.

```json
"annotations": {
  "key1" : "value1",
  "key2" : "value2"
}
```

Possible information that could be recorded in annotations:

* fields managed by a declarative configuration layer, to distinguish them from client- and/or server-set default values and other auto-generated fields, fields set by auto-sizing/auto-scaling systems, etc., in order to facilitate merging
* build/release/image information (timestamps, release ids, git branch, PR numbers, image hashes, registry address, etc.)
* pointers to logging/monitoring/analytics/audit repos
* client library/tool information (e.g. for debugging purposes -- name, version, build info)
* other user and/or tool/system provenance info, such as URLs of related objects from other ecosystem components
* lightweight rollout tool metadata (config and/or checkpoints)
* phone/pager number(s) of person(s) responsible, or directory entry where that info could be found, such as a team website

Yes, this information could be stored in an external database or directory, but that would make it much harder to produce shared client libraries and tools for deployment, management, introspection, etc.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/annotations.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
