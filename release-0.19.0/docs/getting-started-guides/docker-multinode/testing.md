## Testing your Kubernetes cluster.

To validate that your node(s) have been added, run:

```sh
kubectl get nodes
```

That should show something like:
```
NAME           LABELS    STATUS
10.240.99.26   <none>    Ready
127.0.0.1      <none>    Ready
```

If the status of any node is ```Unknown``` or ```NotReady``` your cluster is broken, double check that all containers are running properly, and if all else fails, contact us on IRC at
```#google-containers``` for advice.

### Run an application
```sh
kubectl -s http://localhost:8080 run nginx --image=nginx --port=80
```

now run ```docker ps``` you should see nginx running.  You may need to wait a few minutes for the image to get pulled.

### Expose it as a service:
```sh
kubectl expose rc nginx --port=80
```

This should print:
```
NAME      LABELS    SELECTOR              IP          PORT(S)
nginx     <none>    run=nginx             <ip-addr>   80/TCP
```

Hit the webserver:
```sh
curl <insert-ip-from-above-here>
```

Note that you will need run this curl command on your boot2docker VM if you are running on OS X.

### Scaling 

Now try to scale up the nginx you created before:

```sh
kubectl scale rc nginx --replicas=3
```

And list the pods

```sh
kubectl get pods
```

You should see pods landing on the newly added machine.

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/docker-multinode/testing.md?pixel)]()


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/release-0.19.0/docs/getting-started-guides/docker-multinode/testing.md?pixel)]()
