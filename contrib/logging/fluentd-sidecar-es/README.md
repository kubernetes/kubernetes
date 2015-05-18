# Collecting log files from within containers with Fluentd and sending them to Elasticsearch.
*Note that this only works for clusters with an Elastisearch service. If your cluster is logging to Google Cloud Logging instead (e.g. if you're using Container Engine), see [this guide](/contrib/logging/fluentd-sidecar-gcp/) instead.*

This directory contains the source files needed to make a Docker image that collects log files from arbitrary files within a container using [Fluentd](http://www.fluentd.org/) and sends them to the cluster's Elasticsearch service.
The image is designed to be used as a sidecar container as part of a pod.
It lives in the Google Container Registry under the name `gcr.io/google_containers/fluentd-sidecar-es`.

This shouldn't be necessary if your container writes its logs to stdout or stderr, since the Kubernetes cluster's default logging infrastructure will collect that automatically, but this is useful if your application logs to a specific file in its filesystem and can't easily be changed.

In order to make this work, you have to add a few things to your pod config:

1. A second container, using the `gcr.io/google_containers/fluentd-sidecar-es:1.0` image to send the logs to Elasticsearch.
2. A volume for the two containers to share. The emptyDir volume type is a good choice for this because we only want the volume to exist for the lifetime of the pod.
3. Mount paths for the volume in each container.  In your primary container, this should be the path that the applications log files are written to. In the secondary container, this can be just about anything, so we put it under /mnt/log to keep it out of the way of the rest of the filesystem.
4. The `FILES_TO_COLLECT` environment variable in the sidecar container, telling it which files to collect logs from. These paths should always be in the mounted volume.

To try it out, make sure that your cluster was set up to log to Elasticsearch when it was created (i.e. you set `LOGGING_DESTINATION=elasticsearch`), then simply run
```
kubectl create -f logging-sidecar-pod.yaml
```

You should see the logs show up in the cluster's Kibana log viewer shortly after creating the pod. To clean up after yourself, simply run
```
kubectl delete -f logging-sidecar-pod.yaml
```


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/logging/fluentd-sidecar-es/README.md?pixel)]()
