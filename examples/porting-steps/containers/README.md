<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/examples/porting-steps/containers/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

To Run
------

Install and configure [docker](https://docs.docker.com/installation/).

Run the [mysql official
image](https://registry.hub.docker.com/_/mysql/). Check the [Docker
Hub](https://registry.hub.docker.com/_/mysql/) for details on data
storage. This creates a volume on the host, but does not set it up for
reuse. Optionally follow the [volume
container](https://docs.docker.com/userguide/dockervolumes/) pattern.

```
docker run --name mysql-cont -e MYSQL_ROOT_PASSWORD=mysecretpassword -d mysql
```

The app is written to look at the environment variable `DB_PW` to get
the mysql password. It connects to 'mysql-hostname' to connect to the
database, and expects this to be resolved.

Build and run your app. Link it to the mysql container, pass in the
mysql password.

```
docker build -t twotier .
docker run -d --link mysql-cont:mysql-hostname -p 8080:8080 -e DB_PW=mysecretpassword twotier
```

In your browser, go to [localhost:8080](http://localhost:8080) to see
the app

You could easily run multiple front end containers here. You would
have to broker the ports and figure out load balancing. The command
above uses port 8080 on the host, so no more containers can use that
port.

To shut down, we didn't give the app container a specific name, so
find the name or id with `docker ps` and kill it with `docker rm -f
<name or id>`. The mysql container is named `mysql-cont`, so kill it
with `docker rm -vf mysql-cont`. The `-v` will remove the data volume
and delete the database. If you don't specify `-v` the database will
remain on the host, and disk space will be leaked.




<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/porting-steps/containers/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
