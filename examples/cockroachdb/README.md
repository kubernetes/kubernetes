<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# CockroachDB on Kubernetes as a PetSet

This example deploys [CockroachDB](https://cockroachlabs.com) on Kubernetes as
a PetSet. CockroachDB is a distributed, scalable NewSQL database. Please see
[the homepage](https://cockroachlabs.com) and the
[documentation](https://www.cockroachlabs.com/docs/) for details.

## Limitations

### PetSet limitations

Standard PetSet limitations apply: There is currently no possibility to use
node-local storage (outside of single-node tests), and so there is likely
a performance hit associated with running CockroachDB on some external storage.
Note that CockroachDB already does replication and thus should not be deployed on
a persistent volume which already replicates internally.
High-performance use cases on a private Kubernetes cluster should consider
a DaemonSet deployment.

### Recovery after persistent storage failure

A persistent storage failure (e.g. losing the hard drive) is gracefully handled
by CockroachDB as long as enough replicas survive (two out of three by
default). Due to the bootstrapping in this deployment, a storage failure of the
first node is special in that the administrator must manually prepopulate the
"new" storage medium by running an instance of CockroachDB with the `--join`
parameter. If this is not done, the first node will bootstrap a new cluster,
which will lead to a lot of trouble.

### Dynamic provisioning

The deployment is written for a use case in which dynamic provisioning is
available. When that is not the case, the persistent volume claims need
to be created manually. See [minikube.sh](minikube.sh) for the necessary
steps.

## Testing locally on minikube

Follow the steps in [minikube.sh](minikube.sh) (or simply run that file).

## Simulating failures

When all (or enough) nodes are up, simulate a failure like this:

```shell
kubectl exec cockroachdb-0 -- /bin/bash -c "while true; do kill 1; done"
```

On one of the other pods, run `./cockroach sql --host $(hostname)` and use
(mostly) Postgres-flavor SQL. The example runs with three-fold replication,
so it can tolerate one failure of any given node at a time.
Note also that there is a brief period of time immediately after the creation
of the cluster during which the three-fold replication is established, and
during which killing a node may lead to unavailability.

There is also a [demo script](demo.sh).

## Scaling up or down

Simply edit the PetSet (but note that you may need to create a new persistent
volume claim first). If you ran `minikube.sh`, there's a spare volume so you
can immediately scale up by one. Convince yourself that the new node
immediately serves reads and writes.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/cockroachdb/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
