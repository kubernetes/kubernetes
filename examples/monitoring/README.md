Heapster
===========

Heapster enables monitoring of Kubernetes Clusters using [cAdvisor](https://github.com/google/cadvisor). It currently works only on GCE.

#####Run Heapster in a Kubernetes cluster with an Influxdb backend and [Grafana](http://grafana.org/docs/features/influxdb)

**Step 1: Setup Kube cluster**

Fork the Kubernetes repository and [turn up a Kubernetes cluster](https://github.com/GoogleCloudPlatform/kubernetes-new#contents), if you haven't already. Make sure kubectl.sh is exported.

**Step 2: Start a Pod with Influxdb, grafana and elasticsearch**

```shell
$ kubectl.sh create -f deploy/influx-grafana-pod.json
```

**Step 3: Start Influxdb service**

```shell
$ kubectl.sh create -f deploy/influx-grafana-service.json
```

**Step 4: Update firewall rules**

Open up ports tcp:80,8083,8086,9200.
```shell
$ gcutil addfirewall --allowed=tcp:80,tcp:8083,tcp:8086,tcp:9200 --target_tags=kubernetes-minion heapster
```

**Step 5: Start Heapster Pod**

```shell
$ kubectl.sh create -f deploy/heapster-pod.json
```

Verify that all the pods and services are up and running:

```shell
$ kubectl.sh get pods
```
```shell
$ kubectl.sh get services
```

To start monitoring the cluster using grafana, find out the the external IP of the minion where the 'influx-grafana' Pod is running from the output of `kubectl.sh get pods`, and visit `http://<minion-ip>:80`. 

To access the Influxdb UI visit  `http://<minion-ip>:8083`.

#####Hints
* Grafana's default username and password is 'admin'. You can change that by modifying the grafana container [here](influx-grafana/deploy/grafana-influxdb-pod.json)
* To enable memory and swap accounting on the minions follow the instructions [here](https://docs.docker.com/installation/ubuntulinux/#memory-and-swap-accounting)
