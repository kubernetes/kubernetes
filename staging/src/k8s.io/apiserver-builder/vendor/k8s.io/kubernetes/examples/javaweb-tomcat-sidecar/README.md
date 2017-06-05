## Java Web Application with Tomcat and Sidecar Container

The following document describes the deployment of a Java Web application using Tomcat. Instead of packaging `war` file inside the Tomcat image or mount the `war` as a volume, we use a sidecar container as `war` file provider.

### Prerequisites

https://github.com/kubernetes/kubernetes/blob/master/docs/user-guide/prereqs.md

### Overview

This sidecar mode brings a new workflow for Java users:

![](workflow.png?raw=true "Workflow")

As you can see, user can create a `sample:v2` container as sidecar to "provide" war file to Tomcat by copying it to the shared `emptyDir` volume. And Pod will make sure the two containers compose an "atomic" scheduling unit, which is perfect for this case. Thus, your application version management will be totally separated from web server management.

For example, if you are going to change the configurations of your Tomcat:

```console
$ docker exec -it <tomcat_container_id> /bin/bash
# make some change, and then commit it to a new image
$ docker commit <tomcat_container_id> mytomcat:7.0-dev
```

Done! The new Tomcat image **will not** mess up with your `sample.war` file. You can re-use your tomcat image with lots of different war container images for lots of different apps without having to build lots of different images.

Also this means that rolling out a new Tomcat to patch security or whatever else, doesn't require rebuilding N different images.

**Why not put my `sample.war` in a host dir and mount it to tomcat container?**

You have to **manage the volumes** in this case, for example, when you restart or scale the pod on another node, your contents is not ready on that host.

Generally, we have to set up a distributed file system (NFS at least) volume to solve this (if we do not have GCE PD volume). But this is generally unnecessary.

### How To Set this Up

In Kubernetes a [_Pod_](../../docs/user-guide/pods.md) is the smallest deployable unit that can be created, scheduled, and managed. It's a collocated group of containers that share an IP and storage volume.

Here is the config [javaweb.yaml](javaweb.yaml) for Java Web pod:

NOTE: you should define `war` container **first** as it is the "provider".

<!-- BEGIN MUNGE: javaweb.yaml -->

```
apiVersion: v1
kind: Pod
metadata:
  name: javaweb
spec:
  containers:
  - image: resouer/sample:v1
    name: war
    volumeMounts:
    - mountPath: /app
      name: app-volume
  - image: resouer/mytomcat:7.0
    name: tomcat
    command: ["sh","-c","/root/apache-tomcat-7.0.42-v2/bin/start.sh"]
    volumeMounts:
    - mountPath: /root/apache-tomcat-7.0.42-v2/webapps
      name: app-volume
    ports:
    - containerPort: 8080
      hostPort: 8001
  volumes:
  - name: app-volume
    emptyDir: {}
```

<!-- END MUNGE: EXAMPLE -->

The only magic here is the `resouer/sample:v1` image:

```
FROM busybox:latest
ADD sample.war sample.war
CMD "sh" "mv.sh"
```

And the contents of `mv.sh` is:

```sh
cp /sample.war /app
tail -f /dev/null
```

#### Explanation

1. 'war' container only contains the `war` file of your app
2. 'war' container's CMD tries to copy `sample.war` to the `emptyDir` volume path
3. The last line of `tail -f` is just used to hold the container, as Replication Controller does not support one-off task
4. 'tomcat' container will load the `sample.war` from volume path

What's more, if you don't want to enclose a build-in `mv.sh` script in the `war` container, you can use Pod lifecycle handler to do the copy work, here's a example [javaweb-2.yaml](javaweb-2.yaml):


<!-- BEGIN MUNGE: javaweb-2.yaml -->

```
apiVersion: v1
kind: Pod
metadata:
  name: javaweb-2
spec:
  containers:
  - image: resouer/sample:v2
    name: war
    lifecycle:
      postStart:
        exec:
          command:
            - "cp"
            - "/sample.war"
            - "/app"
    volumeMounts:
    - mountPath: /app
      name: app-volume
  - image: resouer/mytomcat:7.0
    name: tomcat
    command: ["sh","-c","/root/apache-tomcat-7.0.42-v2/bin/start.sh"]
    volumeMounts:
    - mountPath: /root/apache-tomcat-7.0.42-v2/webapps
      name: app-volume
    ports:
    - containerPort: 8080
      hostPort: 8001 
  volumes:
  - name: app-volume
    emptyDir: {}
```

<!-- END MUNGE: EXAMPLE -->

And the `resouer/sample:v2` Dockerfile is quite simple:

```
FROM busybox:latest
ADD sample.war sample.war
CMD "tail" "-f" "/dev/null"
```

#### Explanation

1. 'war' container only contains the `war` file of your app
2. 'war' container's CMD uses `tail -f` to hold the container, nothing more
3. The `postStart` lifecycle handler will do `cp` after the `war` container is started
4. Again 'tomcat' container will load the `sample.war` from volume path

Done! Now your `war` container contains nothing except `sample.war`, clean enough.

### Test It Out

Create the Java web pod:

```console
$ kubectl create -f examples/javaweb-tomcat-sidecar/javaweb-2.yaml
```

Check status of the pod:

```console
$ kubectl get -w po
NAME        READY     STATUS    RESTARTS   AGE
javaweb-2   2/2       Running   0         7s
```

Wait for the status to `2/2` and `Running`. Then you can visit "Hello, World" page on `http://localhost:8001/sample/index.html`

You can also test `javaweb.yaml` in the same way.

### Delete Resources

All resources created in this application can be deleted:

```console
$ kubectl delete -f examples/javaweb-tomcat-sidecar/javaweb-2.yaml
```




<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/javaweb-tomcat-sidecar/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
