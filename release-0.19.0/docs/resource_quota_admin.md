# Administering Resource Quotas

Kubernetes can limit both the number of objects created in a namespace, and the
total amount of resources requested by pods in a namespace.  This facilitates
sharing of a single Kubernetes cluster by several teams or tenants, each in
a namespace.

## Enabling Resource Quota

Resource Quota support is enabled by default for many kubernetes distributions.  It is
enabled when the apiserver `--admission_control=` flag has `ResourceQuota` as
one of its arguments.

Resource Quota is enforced in a particular namespace when there is a
`ResourceQuota` object in that namespace.  There should be at most one
`ResourceQuota` object in a namespace.

##  Object Count Quota
The number of objects of a given type can be restricted.  The following types
are supported:

| ResourceName | Description |
| ------------ | ----------- |
| pods | Total number of pods  |
| services | Total number of services |
| replicationcontrollers | Total number of replication controllers |
| resourcequotas | Total number of resource quotas |
| secrets | Total number of secrets |
| persistentvolumeclaims | Total number of persistent volume claims |

For example, `pods` quota counts and enforces a maximum on the number of `pods`
created in a single namespace.

##  Compute Resource Quota
The total number of objects of a given type can be restricted.  The following types
are supported:

| ResourceName | Description |
| ------------ | ----------- |
| cpu | Total cpu limits of containers |
| memory | Total memory usage limits of containers
| `example.com/customresource` | Total of `resources.limits."example.com/customresource"` of containers |

For example, `cpu` quota sums up the `resources.limits.cpu` fields of every
container of every pod in the namespace, and enforces a maximum on that sum.

Any resource that is not part of core Kubernetes must follow the resource naming convention prescribed by Kubernetes.

This means the resource must have a fully-qualified name (i.e. mycompany.org/shinynewresource)

## Viewing and Setting Quotas
Kubectl supports creating, updating, and viewing quotas
```
$ kubectl namespace myspace
$ cat <<EOF > quota.json
{
  "apiVersion": "v1",
  "kind": "ResourceQuota",
  "metadata": {
    "name": "quota",
  },
  "spec": {
    "hard": {
      "memory": "1Gi",
      "cpu": "20",
      "pods": "10",
      "services": "5",
      "replicationcontrollers":"20",
      "resourcequotas":"1",
    },
  }
}
EOF
$ kubectl create -f quota.json
$ kubectl get quota
NAME
quota
$ kubectl describe quota quota
Name:                   quota
Resource                Used    Hard
--------                ----    ----
cpu                     0m      20
memory                  0       1Gi
pods                    5       10
replicationcontrollers  5       20
resourcequotas          1       1
services                3       5
```

## Quota and Cluster Capacity
Resource Quota objects are independent of the Cluster Capacity.  They are
expressed in absolute units.

Sometimes more complex policies may be desired, such as:
  - proportionally divide total cluster resources among several teams.
  - allow each tenant to grow resource usage as needed, but have a generous
    limit to prevent accidental resource exhaustion.

Such policies could be implemented using ResourceQuota as a building-block, by
writing a controller which watches the quota usage and adjusts the quota
hard limits of each namespace.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/resource_quota_admin.md?pixel)]()


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/release-0.19.0/docs/resource_quota_admin.md?pixel)]()
