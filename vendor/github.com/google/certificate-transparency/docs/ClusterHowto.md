# HOWTO run a clustered CT log server

## Intro
This HOWTO is intended to get you started running a clustered CT "Super Duper" log server.
The scripts in the rep referenced below are targetted at running a log cluster in [Google Compute Engine](https://cloud.google.com)
, but since it's all based around [Docker](https://docker.io) images it shouldn't be too hard to use the scripts as a reference for getting a log running on any other infrastructure where you can easily run Docker containers.


## Prerequisites
You should have:
* Installed the dependencies listed in the top-level 
  [README.md](https://github.com/google/certificate-transparency/README.md) file,
* Installed the Docker utilities (and started the docker daemon if it's not
  already running)
* Signed up for a [Google Cloud](https://cloud.google.com) account _[only
  required if you're intending to use GCE, of course]_

### Dependencies for Debian-based distros
Assuming you're happy to use the stock versions of dependencies, Debian-based
distros (including Ubuntu etc.) have many of them pre-packaged which will
make your life easier, see the main
[README.md](https://github.com/google/certificate-transparency/README.md) for
the canonical list.

In addition to the build dependencies above, you'll need some tools too.
For supported operating systems, the latest versions of these tools can be
installed with the following commands:
```bash
# Install latest version of Docker from docker.io (the system version is very old):
wget -qO- https://get.docker.com/ | sh
# Install latest version of Cloud SDK from Google
curl https://sdk.cloud.google.com | bash
```

## Setup for Google Compute Engine
1. First, create a new project on your GCE [console](https://console.developers.google.com).
1. Put the ID of the project you created into the PROJECT environment variable:
   ```bash
   export PROJECT="your_project_id_here"
   ```
   
1. Enable the following APIs for your new project (look under the `APIs & Auth > APIs` tab in your project's area of your GCE console):
   * Google Cloud APIs > Compute Engine API
   * Google Cloud APIs > Cloud Monitoring API
   * Google Cloud APIs > Compute Engine Instance Groups API
   
1. If you've not done it before, log in with gcloud:
   ```bash
   gcloud auth login
   gcloud config set project ${PROJECT}
   ```
   
## Building Docker images
  Firstly, follow the instructions in the main 
  [README.md](https://github.com/google/certificate-transparency/README.md)
  file to build the C++ code.
  Once that's done, you can move on to building the Docker images by running
  the following commands:

   ```bash
   # Depending on whether you're going to run a Log or a Mirror, you only
   # really need to run one of the next two commands, although running both
   # won't cause any problems.
   sudo docker build -f Dockerfile -t gcr.io/${PROJECT}/super_duper:test . &&
       gcloud docker push gcr.io/${PROJECT}/super_duper:test
   sudo docker build -f Dockerfile-ct-mirror -t gcr.io/${PROJECT}/super_mirror:test . &&
       gcloud docker push gcr.io/${PROJECT}/super_mirror:test
   sudo docker build -f cloud/etcd/Dockerfile -t gcr.io/${PROJECT}/etcd:test . &&
       gcloud docker push gcr.io/${PROJECT}/etcd:test
   sudo docker build -f cloud/prometheus/Dockerfile -t gcr.io/${PROJECT}/prometheus:test . &&
       gcloud docker push gcr.io/${PROJECT}/prometheus:test
   ```

## Starting the cluster on Google Compute Engine
1. Create a new config file for your log/mirror, using one of the existing 
   `cloud/google/configs/<name>.sh` files for reference.
1. Run `cloud/google/create_new_cluster.sh cloud/google/configs/<name>.sh`
1. Make a cup of tea while you wait for the jobs to come up
1. [optional] Create an SSH tunnel for viewing the metrics on Prometheus:
   ```bash
   gcloud compute ssh ${CLUSTER}-prometheus-${ZONE}-1 --ssh-flag="-L 9092:localhost:9090"
   ```
   
   Pointing your browser at [http://localhost:9092](http://localhost:9092)
   should let you add graphs/inspect the metrics gathered by prometheus.


## Stopping the cluster on Google Compute Engine
**WARNING: YOU WILL LOSE THE LOG DATA!**

Run the following commands:
   ```bash
   cloud/google/stop_prometheus.sh
   cloud/google/stop_log.sh
   cloud/google/stop_etcd.sh
   ```

## Updating the log server binaries
If you'd like to update the running CT Log Server code, then it's just a
matter of building & pushing new Docker images (as above), and restarting the
containers.

1. Make whatever changes you wish to the code base, then rebuild and create
   new docker image:
   ```bash
   # For example, to update the image for a CT Log instance:
   make -C cpp -j24 proto/libproto.a server/ct-server 
   sudo docker build -f Dockerfile -t gcr.io/${PROJECT}/super_duper:test . &&
       gcloud docker push gcr.io/${PROJECT}/super_duper:test
   ```
   
1. Use the `cloud/google/update_{log,mirror,prometheus}.sh` scripts to update the jobs:
   ```bash
   # continuing our CT Log theme:
   cloud/google/update_log cloud/google/configs/<name>.sh
   ```

   The containers should re-start with the new image and continue on their way.
   Since the log DB lives on persistent disks mounted by the container images,
   there shouldn't be any data loss doing this.

   The `update_{log,mirror}.sh` scripts will wait for each instance to update
   and become ready to serve again before moving on to update the next
   instance.  This should ensure that you don't lose too much of your cluster's
   serving capacity during the update, but it can mean that logs or mirrors
   which contain a _large_ number of certificates (millions) may take a while
   to update.

   The Log/Mirror binaries will log the git hash from which they were built,
   this can help to verify that they're running the correct version.

## Troubleshooting / Workarounds

### Health Checks on GCE

There seems to be an issue with the way GCE load balancer / health checks are
set up by the scripts that can prevent them working properly. If this happens
the health check status in Networking -> Http Load Balancing ->
mirror-lb-backend is not displayed and the message "Error loading health status"
appears in its place.

This issue is being investigated. As a workaround edit the backend service,
remove the backend mirror groups but don't save yet. Then add all the mirror
groups back again so everything looks like it did before. Now save the edited
service. Wait for a few seconds and the health status should now be displayed
correctly.
