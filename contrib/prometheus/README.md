# Prometheus in Kubernetes

This is an experimental [Prometheus](http://prometheus.io/) setup for monitoring
Kubernetes services that expose prometheus-friendly metrics through address
http://service_address:service_port/metrics.

# Purpose
The purpose of the setup is to gather performance-related metrics during load
tests and analyze them to find and fix bottlenecks.

# Quick start

## Promdash/Prometheus

1. Pick a local directory for promdash.  It can be any directory, preferably one which is stable and which you don't mind keeping around.  Then (in our case, we use */mnt/promdash*, just run this docker command `docker run -v /mnt/promdash:/mnt/promdash -e DATABASE_URL=sqlite3:/mnt/promdash/file.sqlite3 prom/promdash ./bin/rake db:migrate`.  In the future, we might use mysql as the promdash database, however, in any case, this 1 time db setup step is required.

Now quickly confirm that /mnt/promdash/file.sqlite3 exists, and has a non-zero size, and make sure its permissions are open so that containers can read from it.  For example:
```
    [jay@rhbd kubernetes]$ ls -altrh /mnt/promdash/
    total 20K
    drwxr-xr-x. 6 root root 4.0K May  6 23:12 ..
    -rwxrwxrwx  1 root root  12K May  6 23:33 file.sqlite3
```
Looks open enough :).  

1. Now, you can start this pod, like so `kubectl create -f contrib/prometheus/prometheus-all.json`.  This ReplicationController will maintain both prometheus, the server, as well as promdash, the visualization tool.  You can then configure promdash, and next time you restart the pod - you're configuration will be remain (since the promdash directory was mounted as a local docker volume).

1. Finally, you can simply access localhost:3000, which will have promdash running.  Then, add the prometheus server (locahost:9090)to as a promdash server, and create a dashboard according to the promdash directions.

## Prometheus 

You can launch prometheus easily, by simply running.

`kubectl create -f contrib/prometheus/prometheus-all.json`

Then (edit the publicIP field in prometheus-service to be a public ip on one of your kubelets), 

and run 

`kubectl create -f contrib/prometheus/prometheus-service.json`

Now, you can access the service `wget 10.0.1.89:9090`, and build graphs.

## How it works

This is a v1beta3 based, containerized prometheus ReplicationController, which scrapes endpoints which are readable on the KUBERNETES service (the internal kubernetes service running in the default namespace, which is visible to all pods).

1. Use kubectl to handle auth & proxy the kubernetes API locally, emulating the old KUBERNETES_RO service.

1. The list of services to be monitored is passed as a command line aguments in
the yaml file.

1. The startup scripts assumes that each service T will have
2 environment variables set ```T_SERVICE_HOST``` and ```T_SERVICE_PORT``` 

1. Each can be configured manually in yaml file if you want to monitor something
that is not a regular Kubernetes service.  For example, you can add comma delimted
endpoints which can be scraped like so...
```
- -t
- KUBERNETES_RO,MY_OTHER_METRIC_SERVICE
```

# Other notes

For regular Kubernetes services the env variables are set up automatically and injected at runtime. 

By default the metrics are written to a temporary location (that can be changed
in the the volumes section of the yaml file). Prometheus' UI is available 
at port 9090.

# TODO

- We should publish this image into the kube/ namespace.
- Possibly use postgre or mysql as a promdash database.
- push gateway (https://github.com/prometheus/pushgateway) setup.
- stop using kubectl to make a local proxy faking the old RO port and build in
  real auth capabilities.

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/prometheus/README.md?pixel)]()
