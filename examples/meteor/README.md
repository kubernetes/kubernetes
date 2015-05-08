Build a container for your Meteor app
-------------------------------------

To be able to run your Meteor app on Kubernetes you need to build a container for it first. To do that you need to install [Docker](https://www.docker.com) and get an account on [Docker Hub](https://hub.docker.com/). Once you have that you need to add 2 files to your Meteor project "Dockerfile" and ".dockerignore".

"Dockerfile" should contain this:

    FROM chees/meteor-kubernetes
    ENV ROOT_URL http://myawesomeapp.com

You should replace the ROOT_URL with the actual hostname of your app.

The .dockerignore file should contain this:

    .meteor/local
    packages/*/.build*

This tells Docker to ignore the files on those directories when it's building your container.

You can see an example of a Dockerfile in our [meteor-gke-example](https://github.com/Q42/meteor-gke-example) project.

Now you can build your container by running something like this in your Meteor project directory:

    docker build -t chees/meteor-gke-example:1 .

Here you should replace "chees" with your own username on Docker Hub, "meteor-gke-example" with the name of your project and "1" with the version name of your build.

Push the container to your Docker hub account (replace the username and project with your own again):

    docker push chees/meteor-gke-example



Running
-------

Now that you have containerized your Meteor app it's time to set up your cluster. Edit "meteor-controller.json" and make sure the "image" points to the container you just pushed to the Docker Hub.

For Mongo we use a Persistent Disk to store the data. If you're using gcloud you can create it once by running:

    gcloud compute disks create --size=200GB mongo-disk

You also need to format the disk before you can use it:

    gcloud compute instances attach-disk --disk=mongo-disk --device-name temp-data k8s-meteor-master
    gcloud compute ssh k8s-meteor-master --command "sudo mkdir /mnt/tmp && sudo /usr/share/google/safe_format_and_mount /dev/disk/by-id/google-temp-data /mnt/tmp"
    gcloud compute instances detach-disk --disk mongo-disk k8s-meteor-master

Now you can start Mongo using that disk:

    kubectl create -f mongo-pod.json
    kubectl create -f mongo-service.json

Wait until Mongo is started completely and then set up Meteor:

    kubectl create -f meteor-controller.json
    kubectl create -f meteor-service.json

Note that meteor-service.json creates an external load balancer, so your app should be available through the IP of that load balancer once the Meteor pods are started. You can find the IP of your load balancer by running:

    kubectl get services/meteor -o template -t "{{.spec.publicIPs}}"

You might have to open up port 80 if it's not open yet in your project. For example:

    gcloud compute firewall-rules create meteor-80 --allow=tcp:80 --target-tags k8s-meteor-node




TODO replace the mongo image with the official mongo? https://registry.hub.docker.com/_/mongo/

