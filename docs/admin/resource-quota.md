<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Resource Quotas

When several users or teams share a cluster with a fixed number of nodes,
there is a concern that one team could use more than its fair share of resources.

Resource quotas are a tool for administrators to address this concern. Resource quotas
work like this:
- Different teams work in different namespaces.  Currently this is voluntary, but
  support for making this mandatory via ACLs is planned.
- The administrator creates a Resource Quota for each namespace.
- Users put compute resource requests on their pods. The sum of all resource requests across
  all pods in the same namespace must not exceed any hard resource limit in any Resource Quota
  document for the namespace. Note that we used to verify Resource Quota by taking the sum of
  resource limits of the pods, but this was altered to use resource requests. Backwards compatibility
  for those pods previously created is preserved because pods that only specify a resource limit have
  their resource requests defaulted to match their defined limits. The user is only charged for the
  resources they request in the Resource Quota versus their limits because the request is the minimum
  amount of resource guaranteed by the cluster during scheduling. For more information on over commit,
  see [compute-resources](../user-guide/compute-resources.md).
- If creating a pod would cause the namespace to exceed any of the limits specified in the
  the Resource Quota for that namespace, then the request will fail with HTTP status
  code `403 FORBIDDEN`.
- If quota is enabled in a namespace and the user does not specify *requests* on the pod for each
  of the resources for which quota is enabled, then the POST of the pod will fail with HTTP
  status code `403 FORBIDDEN`.  Hint: Use the LimitRange admission controller to force default
  values of *limits* (then resource *requests* would be equal to *limits* by default, see
  [admission controller](admission-controllers.md)) before the quota is checked to avoid this problem.

Examples of policies that could be created using namespaces and quotas are:
- In a cluster with a capacity of 32 GiB RAM, and 16 cores, let team A use 20 Gib and 10 cores,
  let B use 10GiB and 4 cores, and hold 2GiB and 2 cores in reserve for future allocation.
- Limit the "testing" namespace to using 1 core and 1GiB RAM.  Let the "production" namespace
  use any amount.

In the case where the total capacity of the cluster is less than the sum of the quotas of the namespaces,
there may be contention for resources.  This is handled on a first-come-first-served basis.

Neither contention nor changes to quota will affect already-running pods.

## Enabling Resource Quota

Resource Quota support is enabled by default for many Kubernetes distributions.  It is
enabled when the apiserver `--admission-control=` flag has `ResourceQuota` as
one of its arguments.

Resource Quota is enforced in a particular namespace when there is a
`ResourceQuota` object in that namespace.  There should be at most one
`ResourceQuota` object in a namespace.

## Compute Resource Quota

The total sum of [compute resources](../user-guide/compute-resources.md) requested by pods
in a namespace can be limited.  The following compute resource types are supported:

| ResourceName | Description |
| ------------ | ----------- |
| cpu | Total cpu requests of containers |
| memory | Total memory requests of containers

For example, `cpu` quota sums up the `resources.requests.cpu` fields of every
container of every pod in the namespace, and enforces a maximum on that sum.

## Object Count Quota

The number of objects of a given type can be restricted.  The following types
are supported:

| ResourceName | Description |
| ------------ | ----------- |
| pods | Total number of pods  |
| services | Total number of services |
| replicationcontrollers | Total number of replication controllers |
| resourcequotas | Total number of [resource quotas](admission-controllers.md#resourcequota) |
| secrets | Total number of secrets |
| persistentvolumeclaims | Total number of [persistent volume claims](../user-guide/persistent-volumes.md#persistentvolumeclaims) |

For example, `pods` quota counts and enforces a maximum on the number of `pods`
created in a single namespace.

You might want to set a pods quota on a namespace
to avoid the case where a user creates many small pods and exhausts the cluster's
supply of Pod IPs.

## Viewing and Setting Quotas

Kubectl supports creating, updating, and viewing quotas:

```console
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
$ kubectl create -f ./quota.json
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

Resource Quota objects are independent of the Cluster Capacity. They are
expressed in absolute units.  So, if you add nodes to your cluster, this does *not*
automatically give each namespace the ability to consume more resources.

Sometimes more complex policies may be desired, such as:
  - proportionally divide total cluster resources among several teams.
  - allow each tenant to grow resource usage as needed, but have a generous
    limit to prevent accidental resource exhaustion.
  - detect demand from one namespace, add nodes, and increase quota.

Such policies could be implemented using ResourceQuota as a building-block, by
writing a 'controller' which watches the quota usage and adjusts the quota
hard limits of each namespace according to other signals.

Note that resource quota divides up aggregate cluster resources, but it creates no
restrictions around nodes: pods from several namespaces may run on the same node.

## Example

See a [detailed example for how to use resource quota](resourcequota/)..

## Read More

See [ResourceQuota design doc](../design/admission_control_resource_quota.md) for more information.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/resource-quota.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
