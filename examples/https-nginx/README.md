
# Nginx https service

This example creates a basic nginx https service useful in verifying proof of concept, keys, secrets, configmap, and end-to-end https service creation in kubernetes.
It uses an [nginx server block](http://wiki.nginx.org/ServerBlockExample) to serve the index page over both http and https. It will detect changes to nginx's configuration file, default.conf, mounted as a configmap volume and reload nginx automatically.

### Generate certificates

First generate a self signed rsa key and certificate that the server can use for TLS. This step invokes the make_secret.go script in the same directory, which uses the kubernetes api to generate a secret json config in /tmp/secret.json.

```sh
$ make keys secret KEY=/tmp/nginx.key CERT=/tmp/nginx.crt SECRET=/tmp/secret.json
```

### Create a https nginx application running in a kubernetes cluster

You need a [running kubernetes cluster](../../docs/getting-started-guides/) for this to work.

Create a secret and a configmap.

```sh
$ kubectl create -f /tmp/secret.json
secret "nginxsecret" created

$ kubectl create configmap nginxconfigmap --from-file=examples/https-nginx/default.conf
configmap "nginxconfigmap" created
```

Create a service and a replication controller using the configuration in nginx-app.yaml.

```sh
$ kubectl create -f examples/https-nginx/nginx-app.yaml
You have exposed your service on an external port on all nodes in your
cluster.  If you want to expose this service to the external internet, you may
need to set up firewall rules for the service port(s) (tcp:32211,tcp:30028) to serve traffic.
...
service "nginxsvc" created
replicationcontroller "my-nginx" created
```

Then, find the node port that Kubernetes is using for http and https traffic.

```sh
$ kubectl get service nginxsvc -o json
...
                    {
                        "name": "http",
                        "protocol": "TCP",
                        "port": 80,
                        "targetPort": 80,
                        "nodePort": 32211
                    },
                    {
                        "name": "https",
                        "protocol": "TCP",
                        "port": 443,
                        "targetPort": 443,
                        "nodePort": 30028
                    }
...
```

If you are using Kubernetes on a cloud provider, you may need to create cloud firewall rules to serve traffic.
If you are using GCE or GKE, you can use the following commands to add firewall rules.

```sh
$ gcloud compute firewall-rules create allow-nginx-http --allow tcp:32211 --description "Incoming http allowed."
Created [https://www.googleapis.com/compute/v1/projects/hello-world-job/global/firewalls/allow-nginx-http].
NAME              NETWORK  SRC_RANGES  RULES      SRC_TAGS  TARGET_TAGS
allow-nginx-http  default  0.0.0.0/0   tcp:32211

$ gcloud compute firewall-rules create allow-nginx-https --allow tcp:30028 --description "Incoming https allowed."
Created [https://www.googleapis.com/compute/v1/projects/hello-world-job/global/firewalls/allow-nginx-https].
NAME               NETWORK  SRC_RANGES  RULES      SRC_TAGS  TARGET_TAGS
allow-nginx-https  default  0.0.0.0/0   tcp:30028
```

Find your nodes' IPs.

```sh
$ kubectl get nodes -o json | grep ExternalIP -A 2
                        "type": "ExternalIP",
                        "address": "104.198.1.26"
                    }
--
                        "type": "ExternalIP",
                        "address": "104.198.12.158"
                    }
--
                        "type": "ExternalIP",
                        "address": "104.198.11.137"
                    }
```

Now your service is up. You can either use your browser or type the following commands.

```sh
$ curl https://<your-node-ip>:<your-port> -k

$ curl https://104.198.1.26:30028 -k
...
<title>Welcome to nginx!</title>
...
```

Then we will update the configmap by changing `index.html` to `index2.html`.

```sh
kubectl create configmap nginxconfigmap --from-file=examples/https-nginx/default.conf -o yaml --dry-run\
| sed 's/index.html/index2.html/g' | kubectl apply -f -
configmap "nginxconfigmap" configured
```

Wait a few seconds to let the change propagate. Now you should be able to either use your browser or type the following commands to verify Nginx has been reloaded with new configuration.

```sh
$ curl https://<your-node-ip>:<your-port> -k

$ curl https://104.198.1.26:30028 -k
...
<title>Nginx reloaded!</title>
...
```

For more information on how to run this in a kubernetes cluster, please see the [user-guide](../../docs/user-guide/connecting-applications.md).

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/https-nginx/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
