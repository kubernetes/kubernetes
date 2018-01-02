## Running Heapster in Kubernetes with a Google Cloud Monitoring and Google Cloud Logging backend

The Google Cloud Monitoring and Logging backends only works in Google Compute Engine today.

### Setup a Kubernetes cluster
[Bring up a Kubernetes cluster](https://github.com/kubernetes/kubernetes), if you haven't already. Ensure that `kubecfg.sh` is exported.

### Start all of the pods and services

Start the Heapster service on the cluster:

```shell
$ kubectl.sh create -f deploy/kube-config/google/
```

Heapster metrics are now being exported to Google Cloud Monitoring as custom metrics. Heapster events are now being exported to Google Cloud Logging as custom logs.

### Metrics Dashboard
To access the Google Cloud Monitoring dashboard go to: [https://app.google.stackdriver.com/](https://app.google.stackdriver.com/). Create a new dashboard and add the desired charts. Select the *Custom Metric* Resource Type and all Heapster metrics are under the `kubernetes.io` namespace. You can narrow down the query by the metric labels provided.

It is also possible to query the Google Cloud Monitoring data directly using their [custom metric read API](https://cloud.google.com/monitoring/v2beta2/timeseries/list).

### Events Dashboard
To access events via Google Cloud Logging dashboard go to [Google Developer Console](https://cloud.google.com) and select 'Logs' under 'Monitoring' in your project. In the Logs dashboard, select `custom.googleapis.com` as the logs source.
