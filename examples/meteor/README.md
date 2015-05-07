Meteor on Kuberenetes
=====================

This example shows you how to package and run a
[Meteor](https://www.meteor.com/) app on Kubernetes.

Build a container for your Meteor app
-------------------------------------

To be able to run your Meteor app on Kubernetes you need to build a
Docker container for it first. To do that you need to install
[Docker](https://www.docker.com) Once you have that you need to add 2
files to your existing Meteor project `Dockerfile` and
`.dockerignore`.

`Dockerfile` should contain the below lines. You should replace the
`ROOT_URL` with the actual hostname of your app.
```
FROM chees/meteor-kubernetes
ENV ROOT_URL http://myawesomeapp.com
```

The `.dockerignore` file should contain the below lines. This tells
Docker to ignore the files on those directories when it's building
your container.
```
.meteor/local
packages/*/.build*
```

You can see an example meteor project already set up at:
[meteor-gke-example](https://github.com/Q42/meteor-gke-example). Feel
free to use this app for this example.

> Note: The next step will not work if you have added mobile platforms
> to your meteor project. Check with `meteor list-platforms`

Now you can build your container by running this in
your Meteor project directory:
```
docker build -t my-meteor .
```

Pushing to a registry
---------------------

For the [Docker Hub](https://hub.docker.com/), tag your app image with
your username and push to the Hub with the below commands. Replace
`<username>` with your Hub username.
```
docker tag my-meteor <username>/my-meteor
docker push <username>/my-meteor
```

For [Google Container
Registry](https://cloud.google.com/tools/container-registry/), tag
your app image with your project ID, and push to GCR. Replace
`<project>` with your project ID.
```
docker tag my-meteor gcr.io/<project>/my-meteor
gcloud preview docker push gcr.io/<project>/my-meteor
```

Running
-------

Now that you have containerized your Meteor app it's time to set up
your cluster. Edit `meteor-controller.json` and make sure the `image`
points to the container you just pushed to the Docker Hub or GCR.

As you may know, Meteor uses MongoDB, and we'll need to provide it a
persistant Kuberetes volume to store its data. See the [volumes
documentation](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/volumes.md)
for options. We're going to use Google Compute Engine persistant
disks. Create the MongoDB disk by running:
```
gcloud compute disks create --size=200GB mongo-disk
```

You also need to format the disk before you can use it:
```
gcloud compute instances attach-disk --disk=mongo-disk --device-name temp-data kubernetes-master
gcloud compute ssh kubernetes-master --command "sudo mkdir /mnt/tmp && sudo /usr/share/google/safe_format_and_mount /dev/disk/by-id/google-temp-data /mnt/tmp"
gcloud compute instances detach-disk --disk mongo-disk kubernetes-master
```

Now you can start Mongo using that disk:
```
kubectl create -f mongo-pod.json
kubectl create -f mongo-service.json
```

Wait until Mongo is started completely and then start up your Meteor app:
```
kubectl create -f meteor-controller.json
kubectl create -f meteor-service.json
```

Note that `meteor-service.json` creates an external load balancer, so
your app should be available through the IP of that load balancer once
the Meteor pods are started. You can find the IP of your load balancer
by running:
```
kubectl get services/meteor -o template -t "{{.spec.publicIPs}}"
```

You will have to open up port 80 if it's not open yet in your
environment. On GCE, you may run the below command.
```
gcloud compute firewall-rules create meteor-80 --allow=tcp:80 --target-tags kubernetes-minion
```

What is going on?
-----------------

Firstly, the `FROM chees/meteor-kubernetes` line in your `Dockerfile`
specifies the base image for your Meteor app. The code for that image
is located in the `dockerbase/` subdirectory. Open up the `Dockerfile`
to get an insight of what happens during the `docker build` step. The
image is based on the Node.js official image. It then installs Meteor
and copies in your apps' code. The last line specifies what happens
when your app container is run.
```
ENTRYPOINT MONGO_URL=mongodb://$MONGO_SERVICE_HOST:$MONGO_SERVICE_PORT /usr/local/bin/node main.js
```

Here we can see the MongoDB host and port information being passed
into the Meteor app. The `MONGO_SERVICE...` environment variables are
set by Kubernetes, and point to the service named `mongo` specified in
`mongo-service.json`. See the [environment
docuementation](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/container-environment.md)
for more details.

As you may know, Meteor uses long lasting connections, and requires
_sticky sessions_. With Kubernetes you can scale out your app easily
with session affinity. The `meteor-service.json` file contains
`"sessionAffinity": "ClientIP"`, which provides this for us. See the
[service
documentation](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/services.md#portals-and-service-proxies)
for more information.

As mentioned above, the mongo container uses a volume which is mapped
to a persistant disk by Kubernetes. In `mongo-pod.json` the container
section specifies the volume:
```
        "volumeMounts": [
          {
            "name": "mongo-disk",
            "mountPath": "/data/db"
          }
```

The name `mongo-disk` refers to the volume specified outside the
container section:
```
    "volumes": [
      {
        "name": "mongo-disk",
        "gcePersistentDisk": {
          "pdName": "mongo-disk",
          "fsType": "ext4"
        }
      }
    ],
```
