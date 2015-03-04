# API Change Guidelines

Freely change pkg/api/types.go. 

Update the most recent versioned package's conversion functions. (In, e.g., pkg/api/v1beta1/conversion.go)

If necessary, you may add fields to the versioned package's objects. You may not remove fields or rename them (or change the json/yaml tags), because that will break clients.

This allows us to make progress on the internal API without breaking clients. pkg/api/serialization_test.go should detect errors that will result in information being lost as a result of doing something wrong in modifying the API.

At some point, we may want to cut a new version. Assume the new version is called v8 and the previous version is called v7:

1. Copy what we have in pkg/api/types.go to pkg/api/v8/types.go.
2. Fix up links in pkg/api/v7's conversion package-- now they convert to pkg/api/v8 and not pkg/api.
3. pkg/api/v8 gets a new register.go and a conversion.go, the latter having nothing to do yet.
4. Update pkg/api/latest to use the new version as the default.
5. Update RAML documentation. (We really need to start auto generating it...)

(Note that this info may need to change when we finish cutting v1beta2.)

Q (Clayton): what are we going to do about old clients that PUT old API versions that lack a field in the new internal?  Let them clobber it?  Intelligently merge once we get the conflicting version from etcd?  Seems like we need to know what version of the API the client submitted a result in to know whether to merge a field or not.
