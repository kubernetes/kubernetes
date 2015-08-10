# Bare Metal Service Load Balancers

AKA "how to set up a bank of haproxy for platforms that don't have load balancers".

## Disclaimer:
- This is a **work in progress**.
- A better way to achieve this will probably emerge once discussions on (#260, #561) converge.
- Backends are pluggable, but [Haproxy](https://cbonte.github.io/haproxy-dconv/configuration-1.5.html) is the only loadbalancer with a working implementation.
- I have never deployed haproxy to production, so contributions are welcome (see [wishlist](#wishlist) for ideas).
- For fault tolerant load balancing of ingress traffic, you need:
  1. Multiple hosts running load balancers
  2. Multiple A records for each hostname in a DNS service.

This module will not help with the latter

## Overview

### Ingress

There are 2 ways to expose a service to ingress traffic in the current kubernetes service model:

- Create a cloud load balancer.
- Allocate a port (the same port) on every node in your cluster and proxy ingress traffic through that port to the endpoints.

The service-loadbalancer aims to give you 1 on bare metal, making 2 unnecessary for the common case. The replication controller manifest in this directly creates a service-loadbalancer pod on all nodes with the `role=loadbalancer` label. Each service-loadbalancer pod contains:
- A load balancer controller that watches the kubernetes api for services and endpoints.
- A load balancer manifest. This is used to bootstrap the load balancer. The load balancer itself is pluggable, so you can easily swap
  haproxy for something like [f5](https://f5.com/glossary/load-balancer) or [pound](http://www.apsis.ch/pound).
- A template used to write load balancer rules. This is tied to the loadbalancer used in the manifest, since each one has a different config format.

__L7 load balancing of Http services__: The load balancer controller automatically exposes http services to ingress traffic on all nodes with a `role=loadbalancer` label. It assumes all services are http unless otherwise instructed. Each http service gets a loadbalancer forwarding rule, such that requests received on `http://loadbalancer-node/serviceName:port` balanced between its endpoints according to the algorithm specified in the loadbalacer.json manifest. You do not need more than a single loadbalancer pod to balance across all your http services (you can scale the rc to increase capacity).

__L4 loadbalancing of Tcp services__: Since one needs to specify ports at pod creation time (kubernetes doesn't currently support port ranges), a single loadbalancer is tied to a set of preconfigured node ports, and hence a set of TCP services it can expose. The load balancer controller will dynamically add rules for each configured TCP service as it pops into existence. However, each "new" (unspecified in the tcpServices section of the loadbalancer.json) service will need you to open up a new container-host port pair for traffic. You can achieve this by creating a new loadbalancer pod with the `targetPort` set to the name of your service, and that service specified in the tcpServices map of the new loadbalancer.

### Cross-cluster loadbalancing

On cloud providers that offer a private ip range for all instances on a network, you can setup multiple clusters in different availability zones, on the same network, and loadbalancer services across these zones. On GCE for example, every instance is a member of a single network. A network performs the same function that a router does: it defines the network range and gateway IP address, handles communication between instances, and serves as a gateway between instances and other networks. On such networks the endpoints of a service in one cluster are visible in all other clusters in the same network, so you can setup an edge loadbalancer that watches a kubernetes master of another cluster for services. Such a deployment allows you to fallback to a different AZ during times of duress or planned downtime (eg: database update).

### Examples

Initial cluster state:
```console
$ kubectl get svc --all-namespaces -o yaml  | grep -i "selfLink"
    selfLink: /api/v1/namespaces/default/services/kubernetes
    selfLink: /api/v1/namespaces/default/services/nginxsvc
    selfLink: /api/v1/namespaces/kube-system/services/elasticsearch-logging
    selfLink: /api/v1/namespaces/kube-system/services/kibana-logging
    selfLink: /api/v1/namespaces/kube-system/services/kube-dns
    selfLink: /api/v1/namespaces/kube-system/services/kube-ui
    selfLink: /api/v1/namespaces/kube-system/services/monitoring-grafana
    selfLink: /api/v1/namespaces/kube-system/services/monitoring-heapster
    selfLink: /api/v1/namespaces/kube-system/services/monitoring-influxdb
```
These are all the [cluster addon](../../cluster/addons) services in `namespace=kube-system`.

#### Create a loadbalancer
* Loadbalancers are created via a ReplicationController.
* Load balancers will only run on nodes with the `role=loadbalancer` label.
```console
$ kubectl create -f ./rc.yaml
replicationcontrollers/service-loadbalancer
$  kubectl get pods -l app=service-loadbalancer
NAME                         READY     STATUS    RESTARTS   AGE
service-loadbalancer-dapxv   0/2       Pending   0          1m
$ kubectl describe pods -l app=service-loadbalancer
Events:
  FirstSeen                                    From            Reason                  Message
  Tue, 21 Jul 2015 11:19:22 -0700              {scheduler }    failedScheduling        Failed for reason MatchNodeSelector and possibly others
```

Notice that the pod hasn't started because the scheduler is waiting for you to tell it which nodes to use as a load balancer.

```console
$ kubectl label node e2e-test-beeps-minion-c9up role=loadbalancer
NAME                         LABELS                                                                STATUS
e2e-test-beeps-minion-c9up   kubernetes.io/hostname=e2e-test-beeps-minion-c9up,role=loadbalancer   Ready
```
#### Expose services
Your kube-ui should be publicly accessible once the loadbalancer created in the previous step is in `Running` (if you're on a cloud provider, you need to create firewall-rules for :80)
```console
$ kubectl get nodes e2e-test-beeps-minion-c9up -o json | grep -i externalip -A 1
                "type": "ExternalIP",
                "address": "104.197.63.17"
$ curl http://104.197.63.17/kube-ui
```

#### HTTP
You can use the [https-nginx](../../examples/https-nginx) example to create some new HTTP/HTTPS services.

```console
$ cd ../../examples/https-nginx
$ make keys secret KEY=/tmp/nginx.key CERT=/tmp/nginx.crt SECRET=/tmp/secret.json
$ kubectl create -f /tmp/secret.json
$ kubectl get secrets
NAME                  TYPE                                  DATA
default-token-vklfs   kubernetes.io/service-account-token   2
nginxsecret           Opaque                                2
```

Lets introduce a small twist. The nginx-app example exposes the nginx service using `NodePort`, which means it opens up a random port on every node in your cluster and exposes the service on that. Delete the `type: NodePort` line before creating it.

```console
$ kubectl create -f nginx-app.yaml
$ kubectl get svc
NAME         LABELS                                    SELECTOR    IP(S)         PORT(S)
kubernetes   component=apiserver,provider=kubernetes   <none>      10.0.0.1      443/TCP
nginxsvc     app=nginx                                 app=nginx   10.0.79.131   80/TCP
                                                                                 443/TCP
$ curl http://104.197.63.17/nginxsvc
```

#### HTTPS
HTTPS services are handled at L4 (see [wishlist](#wishlist))
```console
$ curl https://104.197.63.17:8080 -k
```

A couple of points to note:
- The nginxsvc is specified in the tcpServices of the loadbalancer.json manifest.
- The https service is accessible directly on the specified port, which matches the *service port*.
- You need to take care of ensuring there is no collision between these service ports on the node.

#### TCP

```yaml
$ cat mysql-app.yaml
apiVersion: v1
kind: Pod
metadata:
  name: mysql
  labels:
    name: mysql
spec:
  containers:
  - image: mysql
    name: mysql
    env:
    - name: MYSQL_ROOT_PASSWORD
      # Use secrets instead of env for passwords
      value: password
    ports:
    - containerPort: 3306
      name: mysql
    volumeMounts:
    # name must match the volume name below
    - name: mysql-storage
      # mount path within the container
      mountPath: /var/lib/mysql
  volumes:
  - name: mysql-storage
    emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  labels:
    name: mysql
  name: mysql
spec:
  type: NodePort
  ports:
    # the port that this service should serve on
    - port: 3306
  # label keys and values that must match in order to receive traffic for this service
  selector:
    name: mysql
```

We'll create the service and access mysql from outside the cluster:
```console
$ kubectl create -f mysql-app.yaml
$ kubeclt get svc
NAME         LABELS                                    SELECTOR    IP(S)         PORT(S)
kubernetes   component=apiserver,provider=kubernetes   <none>      10.0.0.1      443/TCP
nginxsvc     app=nginx                                 app=nginx   10.0.79.131   80/TCP
                                                                                 443/TCP
mysql        app=mysql                                 app=mysql   10.0.63.72    3306/TCP

$ mysql -u root -ppassword --host 104.197.63.17 --port 3306 -e 'show databases;'
+--------------------+
| Database           |
+--------------------+
| information_schema |
| mysql              |
| performance_schema |
+--------------------+
```


#### Cross-cluster loadbalancing

First setup your 2 clusters, and a kubeconfig secret as described in the [sharing clusters example] (../../examples/sharing-clusters/README.md). We will create a loadbalancer in our first cluster (US) and have it publish the services from the second cluster (EU). This is the entire modified loadbalancer manifest:

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: service-loadbalancer
  labels:
    app: service-loadbalancer
    version: v1
spec:
  replicas: 1
  selector:
    app: service-loadbalancer
    version: v1
  template:
    metadata:
      labels:
        app: service-loadbalancer
        version: v1
    spec:
      volumes:
      # token from the eu cluster, must already exist
      # and match the name of the volume using in container
      - name: eu-config
        secret:
          secretName: kubeconfig
      nodeSelector:
        role: loadbalancer
      containers:
      - image: gcr.io/google_containers/servicelb:0.1
        imagePullPolicy: Always
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8081
            scheme: HTTP
          initialDelaySeconds: 30
          timeoutSeconds: 5
        name: haproxy
        ports:
        # All http services
        - containerPort: 80
          hostPort: 80
          protocol: TCP
        # nginx https
        - containerPort: 443
          hostPort: 8080
          protocol: TCP
        # mysql
        - containerPort: 3306
          hostPort: 3306
          protocol: TCP
        # haproxy stats
        - containerPort: 1936
          hostPort: 1936
          protocol: TCP
        resources: {}
        args:
        - --tcp-services=mysql:3306,nginxsvc:443
        - --use-kubernetes-cluster-service=false
        # use-kubernetes-cluster-service=false in conjunction with the
        # kube/config will force the service-loadbalancer to watch for
        # services form the eu cluster.
        volumeMounts:
        - mountPath: /.kube
          name: eu-config
        env:
        - name: KUBECONFIG
          value: /.kube/config
```

Note that it is essentially the same as the rc.yaml checked into the service-loadbalancer directory expect that it consumes the kubeconfig secret as an extra KUBECONFIG environment variable.

```cmd
$ kubectl config use-context <us-clustername>
$ kubectl create -f rc.yaml
$ kubectl get pods -o wide
service-loadbalancer-5o2p4   1/1       Running   0          13m       kubernetes-minion-5jtd
$ kubectl get node kubernetes-minion-5jtd -o json | grep -i externalip -A 2
                "type": "ExternalIP",
                "address": "104.197.81.116"
$ curl http://104.197.81.116/nginxsvc
Europe
```

### Troubleshooting:
- If you can curl or netcat the endpoint from the pod (with kubectl exec) and not from the node, you have not specified hostport and containerport.
- If you can hit the ips from the node but not from your machine outside the cluster, you have not opened firewall rules for the right network.
- If you can't hit the ips from within the container, either haproxy or the service_loadbalacer script is not running.
  1. Use ps in the pod
  2. sudo restart haproxy in the pod
  3. cat /etc/haproxy/haproxy.cfg in the pod
  4. try kubectl logs haproxy
  5. run the service_loadbalancer with --dry
- Check http://<node_ip>:1936 for the stats page. It requires the password used in the template file.
- Try talking to haproxy on the stats socket directly on the container using kubectl exec, eg: echo “show info” | socat unix-connect:/tmp/haproxy stdio

### Wishlist:

- Allow services to specify their url routes (see [openshift routes](https://github.com/openshift/origin/blob/master/docs/routing.md))
- Scrape :1926 and scale replica count of the loadbalancer rc from a helper pod (this is basically ELB)
- Scrape :1936/;csv and autoscale services
- Better https support. 3 options to handle ssl:
  1. __Termination__: certificate lives on load balancer. All traffic to load balancer is encrypted, traffic from load balancer to service is not.
  2. __Pass Through__: Load balancer drops down to L4 balancing and forwards TCP encrypted packets to destination.
  3. __Redirect__: All traffic is https. HTTP connections are encrypted using load balancer certs.

  Currently you need to trigger TCP loadbalancing for your https service by specifying it in loadbalancer.json. Support for the other 2 would be nice.
- Multinamespace support: Currently the controller only watches a single namespace for services.
- Support for external services (eg: amazon rds)
- Dynamically modify loadbalancer.json. Will become unnecessary when we have a loadbalancer resource.
- Headless services: I just didn't think people would care enough about this.



[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/service-loadbalancer/README.md?pixel)]()
