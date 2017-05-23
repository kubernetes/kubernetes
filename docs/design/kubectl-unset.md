# Kubectl Unset Subcommand

## Goals

`kubectl unset` is used to remove specific features from objects. To be contrasted with `kubectl set` subcommand, `unset` commands should has the same ability as `set` commands.

## Design

The `unset` subcommand helps users modify the existing application resources. Using `kubectl unset Object Resource` to remove features from `Object` of `Resource`(e.g. `kubectl unset subject rolebinding`).

### Generic Code Template

Now that `unset` subcmmand is highly similar to `set` subcommand, we could design a generic code template to reuse code.

In set_object.go, the part to change object should be separated from options.Run(), and we should make it as a callback funtion for options.Run().

For instance, we could pass setObjects() into options.Run(setObjects) in set_object.go. In the other side, we could create unsetObjects() as a callback funtion in unset_object.go.

## Examples

Create a rolebinding.
```
$ ./cluster/kubectl.sh create rolebinding foo --user=user1 --group=group1 --role=foo
rolebinding "foo" created
$ ./cluster/kubectl.sh get rolebinding foo -o yaml
apiVersion: rbac.authorization.k8s.io/v1alpha1
kind: RoleBinding
metadata:
  creationTimestamp: 2017-05-23T06:53:07Z
  name: foo
  namespace: default
  resourceVersion: "1082"
  selfLink: /apis/rbac.authorization.k8s.io/v1alpha1/namespaces/default/rolebindings/foo
  uid: 7a16a105-3f84-11e7-8292-7427ea6f0fe3
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: foo
subjects:
- apiVersion: rbac.authorization.k8s.io/v1alpha1
  kind: User
  name: user1
- apiVersion: rbac.authorization.k8s.io/v1alpha1
  kind: Group
  name: group1
```

Remove user1 from rolebinding.
```
$ ./cluster/kubectl.sh unset subject rolebinding foo --user=user1
rolebinding "foo" subjects updated
$ ./cluster/kubectl.sh get rolebinding foo -o yaml
apiVersion: rbac.authorization.k8s.io/v1alpha1
kind: RoleBinding
metadata:
  creationTimestamp: 2017-05-23T06:53:07Z
  name: foo
  namespace: default
  resourceVersion: "1107"
  selfLink: /apis/rbac.authorization.k8s.io/v1alpha1/namespaces/default/rolebindings/foo
  uid: 7a16a105-3f84-11e7-8292-7427ea6f0fe3
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: foo
subjects:
- apiVersion: rbac.authorization.k8s.io/v1alpha1
  kind: Group
  name: group1
```
