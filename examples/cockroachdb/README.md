# CockroachDB on Kubernetes as a StatefulSet

This example deploys [CockroachDB](https://cockroachlabs.com) on Kubernetes as
a StatefulSet. CockroachDB is a distributed, scalable NewSQL database. Please see
[the homepage](https://cockroachlabs.com) and the
[documentation](https://www.cockroachlabs.com/docs/) for details.

## Limitations

### StatefulSet limitations

Standard StatefulSet limitations apply: There is currently no possibility to use
node-local storage (outside of single-node tests), and so there is likely
a performance hit associated with running CockroachDB on some external storage.
Note that CockroachDB already does replication and thus it is unnecessary to
deploy it onto persistent volumes which already replicate internally.
For this reason, high-performance use cases on a private Kubernetes cluster
may want to consider a DaemonSet deployment until Stateful Sets support node-local
storage (see #7562).

### Recovery after persistent storage failure

A persistent storage failure (e.g. losing the hard drive) is gracefully handled
by CockroachDB as long as enough replicas survive (two out of three by
default). Due to the bootstrapping in this deployment, a storage failure of the
first node is special in that the administrator must manually prepopulate the
"new" storage medium by running an instance of CockroachDB with the `--join`
parameter. If this is not done, the first node will bootstrap a new cluster,
which will lead to a lot of trouble.

### Dynamic volume provisioning

The deployment is written for a use case in which dynamic volume provisioning is
available. When that is not the case, the persistent volume claims need
to be created manually. See [minikube.sh](minikube.sh) for the necessary
steps. If you're on GCE or AWS, where dynamic provisioning is supported, no
manual work is needed to create the persistent volumes.

## Testing locally on minikube

Follow the steps in [minikube.sh](minikube.sh) (or simply run that file).

## Testing in the cloud on GCE or AWS

Once you have a Kubernetes cluster running, just run
`kubectl create -f cockroachdb-statefulset.yaml` to create your cockroachdb cluster.
This works because GCE and AWS support dynamic volume provisioning by default,
so persistent volumes will be created for the CockroachDB pods as needed.

## Accessing the database

Along with our StatefulSet configuration, we expose a standard Kubernetes service
that offers a load-balanced virtual IP for clients to access the database
with. In our example, we've called this service `cockroachdb-public`.

Start up a client pod and open up an interactive, (mostly) Postgres-flavor
SQL shell using:

```console
$ kubectl run -it --rm cockroach-client --image=cockroachdb/cockroach --restart=Never --command -- ./cockroach sql --host cockroachdb-public
```

You can see example SQL statements for inserting and querying data in the
included [demo script](demo.sh), but can use almost any Postgres-style SQL
commands. Some more basic examples can be found within
[CockroachDB's documentation](https://www.cockroachlabs.com/docs/learn-cockroachdb-sql.html).

## Accessing the admin UI

If you want to see information about how the cluster is doing, you can try
pulling up the CockroachDB admin UI by port-forwarding from your local machine
to one of the pods:

```shell
kubectl port-forward cockroachdb-0 8080
```

Once you’ve done that, you should be able to access the admin UI by visiting
http://localhost:8080/ in your web browser.

## Simulating failures

When all (or enough) nodes are up, simulate a failure like this:

```shell
kubectl exec cockroachdb-0 -- /bin/bash -c "while true; do kill 1; done"
```

You can then reconnect to the database as demonstrated above and verify
that no data was lost. The example runs with three-fold replication, so
it can tolerate one failure of any given node at a time. Note also that
there is a brief period of time immediately after the creation of the
cluster during which the three-fold replication is established, and during
which killing a node may lead to unavailability.

The [demo script](demo.sh) gives an example of killing one instance of the
database and ensuring the other replicas have all data that was written.

## Scaling up or down

Simply patch the Stateful Set by running

```shell
kubectl patch statefulset cockroachdb -p '{"spec":{"replicas":4}}'
```

Note that you may need to create a new persistent volume claim first. If you
ran `minikube.sh`, there's a spare volume so you can immediately scale up by
one. If you're running on GCE or AWS, you can scale up by as many as you want
because new volumes will automatically be created for you. Convince yourself
that the new node immediately serves reads and writes.

## Cleaning up when you're done

Because all of the resources in this example have been tagged with the label `app=cockroachdb`,
we can clean up everything that we created in one quick command using a selector on that label:

```shell
kubectl delete statefulsets,pods,persistentvolumes,persistentvolumeclaims,services -l app=cockroachdb
```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/cockroachdb/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
