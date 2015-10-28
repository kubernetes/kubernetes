<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Working with Resources

*This document is aimed at users who have worked through some of the examples,
and who want to learn more about using kubectl to manage resources such
as pods and services.  Users who want to access the REST API directly,
and developers who want to extend the Kubernetes API should
refer to the [api conventions](../devel/api-conventions.md) and
the [api document](../api.md).*

## Resources are Automatically Modified

When you create a resource such as pod, and then retrieve the created
resource, a number of the fields of the resource are added.
You can see this at work in the following example:

```console
$ cat > /tmp/original.yaml <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: foo
spec:
  containers:
    - name: foo
      image: busybox
  restartPolicy: Never
EOF
$ kubectl create -f /tmp/original.yaml
pods/original
$ kubectl get pods/original -o yaml > /tmp/current.yaml
pods/original
$ wc -l /tmp/original.yaml /tmp/current.yaml
      51 /tmp/current.yaml
       9 /tmp/original.yaml
      60 total
```

The resource we posted had only 9 lines, but the one we got back had 51 lines.
If you `diff -u /tmp/original.yaml /tmp/current.yaml`, you can see the fields added to the pod.
The system adds fields in several ways:
  - Some fields are added synchronously with creation of the resource and some are set asynchronously.
    - For example: `metadata.uid` is set synchronously.  (Read more about [metadata](../devel/api-conventions.md#metadata)).
    - For example, `status.hostIP` is set only after the pod has been scheduled.  This often happens fast, but you may notice pods which do not have this set yet.  This is called Late Initialization.  (Read mode about [status](../devel/api-conventions.md#spec-and-status) and [late initialization](../devel/api-conventions.md#late-initialization) ).
  - Some fields are set to default values.  Some defaults vary by cluster and some are fixed for the API at a certain version.  (Read more about [defaulting](../devel/api-conventions.md#defaulting)).
    - For example, `spec.containers[0].imagePullPolicy` always defaults to `IfNotPresent` in api v1.
    - For example, `spec.containers[0].resources.limits.cpu` may be defaulted to  `100m` on some clusters, to some other value on others, and not defaulted at all on others.
The API will generally not modify fields that you have set; it just sets ones which were unspecified.

## <a name="finding_schema_docs"></a>Finding Documentation on Resource Fields

You can browse auto-generated API documentation at the [project website](http://kubernetes.io/v1.1/api-ref.html) or on [github](../../docs/api-reference/).




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/working-with-resources.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
