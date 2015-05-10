# Downward API

The downward API allows containers to consume information about the system without coupling to the
kubernetes client or REST API.

### Capabilities

Containers can consume the following information via the downward API:

*   Their pod's name
*   Their pod's namespace

### Consuming information about a pod in a container

Containers consume information from the downward API using environment variables.  In the future,
containers will also be able to consume the downward API via a volume plugin.  The `valueFrom`
field of an environment variable allows you to specify an `ObjectFieldSelector` to select fields
from the pod's definition.  The `ObjectFieldSelector` has an `apiVersion` field and a `fieldPath`
field.  The `fieldPath` field is an expression designating a field on the pod.  The `apiVersion`
field is the version of the API schema that the `fieldPath` is written in terms of.  If the
`apiVersion` field is not specified it is defaulted to the API version of the enclosing object.

### Example: consuming the downward API

This is an example of a pod that consumes its name and namespace via the downward API:

```yaml
apiVersion: v1beta3
kind: Pod
metadata:
  name: dapi-test-pod
spec:
  containers:
    - name: test-container
      image: gcr.io/google_containers/busybox
      command: [ "/bin/sh", "-c", "env" ]
      env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
  restartPolicy: Never
```
