## Status v1 meta

Group        | Version     | Kind
------------ | ---------- | -----------
meta | v1 | Status



Status is a return value for calls that don't return other objects.



Field        | Description
------------ | -----------
apiVersion <br /> *string*    | APIVersion defines the versioned schema of this representation of an object. Servers should convert recognized schemas to the latest internal value, and may reject unrecognized values. More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#resources
code <br /> *integer*    | Suggested HTTP return code for this status, 0 if not set.
details <br /> *[StatusDetails](#statusdetails-v1)*    | Extended data associated with the reason.  Each reason may define its own extended details. This field is optional and the data returned is not guaranteed to conform to any schema except that defined by the reason type.
kind <br /> *string*    | Kind is a string value representing the REST resource this object represents. Servers may infer this from the endpoint the client submits requests to. Cannot be updated. In CamelCase. More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
message <br /> *string*    | A human-readable description of the status of this operation.
metadata <br /> *[ListMeta](#listmeta-v1)*    | Standard list metadata. More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#types-kinds
reason <br /> *string*    | A machine-readable description of why this operation is in the "Failure" status. If this value is empty there is no information available. A Reason clarifies an HTTP status code but does not override it.
status <br /> *string*    | Status of the operation. One of: "Success" or "Failure". More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status

