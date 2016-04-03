# Run Heapster in a Kubernetes cluster with an InfluxDB backend and a Grafana UI

### Setup a Kubernetes cluster
[Bring up a Kubernetes cluster](https://github.com/kubernetes/kubernetes), if you haven't already. Ensure that `kubecfg.sh` is exported.
The cluster must have the kube proxy and kube-dns running.

### Start all of the pods and services
```shell
$ kubectl create -f deploy/kube-config/influxdb/
```

Grafana service by default using the api-proxy to relay requests to Grafana. It can then be accessed at http://<KUBE-API-IP>:8080/api/v1/proxy/namespaces/kube-system/services/monitoring-grafana/


For production use, configure a LoadBalancer or if that is not available in your cluster, consider using a NodePort. Use the external IP assigned to the Grafana service, to access Grafana.

The default user name and password is 'admin' but you shouldn't need to login to access Grafana. In Grafana you should find a datasource already configured called InfluxDB. The URL for InfluxDB will be `http://localhost:8086`. Database name is 'k8s'. Default user name and password is 'root'. 
Grafana documentation for InfluxDB [here](http://docs.grafana.org/datasources/influxdb/).

Take a look at the [storage schema](storage-schema.md) to understand how metrics are stored in InfluxDB.

Grafana is set up to auto-populate nodes and pods using templates.

## Troubleshooting guide
1. If the Grafana service is not accessible, chances are it might not be running. Use `kubectl` to verify that the `heapster` and `influxdb & grafana` pods are alive.

	kubectl get pods --all-namespaces

	kubectl get services --all-namespaces

2. To access the InfluxDB UI, you will have to make the InfluxDB service externally visible, similar to how Grafana is made publicly accessible.

3. If you find InfluxDB to be using up a lot of CPU or memory, consider placing resource restrictions on the `InfluxDB & Grafana` pod. You can add `cpu: <millicores>` and `memory: <bytes>` in the [Controller Spec](../deploy/kube-config/influxdb/influxdb-grafana-controller.yaml) and relaunch the controllers:
