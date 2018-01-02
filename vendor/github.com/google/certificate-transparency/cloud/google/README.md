Running on Google Cloud
=======================

The scripts and configs here enable you to run CT-Log and -Mirror instances on
Google's Cloud.

You'll need a Google Cloud enabled account, which you can sign up for at
(cloud.google.com)[https://cloud.google.com], and you should configure your billing settings.

How to run a mirror instance on GCE
-----------------------------------

1. Create a new project on your cloud console, e.g. `ct-mirror-<target>`
1. Set-up APIs/Monitoring:
   Click on `APIs`, and enable the following APIs:
   1. `Compute Engine` (click on compute engine, then `Enable API`)
   1. `Cloud Storage`
   1. `Cloud Monitoring`
   1. `Compute Engine Instance Groups`
1. Click on `Monitoring > Dashboards & alerts`
   Click on `Enable Monitoring`
1. Create a config file for your mirror (see examples under [cloud/google/configs](https://github.com/google/certificate-transparency/cloud/google/configs).
   This config defines a mirror for Google's Pilot log, with two mirror
   instances running in each of the 3 zones in the `us-central1` region (see
   (here)[https://cloud.google.com/compute/docs/zones] for more information
   on regions and zones, and how to choose them.)  Monitoring is configured to
   use Google Cloud Monitoring.

   ```bash
   PROJECT="my-project"
   INSTANCE_TYPE="mirror"
   CLUSTER="my-pilot-mirror"
   REGION="us-central1" # run "gcloud compute regions list" for a list to choose from
   ZONES="a b c"       # run "gcloud compute zones list" for a list to choose from
   MIRROR_TARGET_URL="https://ct.googleapis.com/pilot"
   MIRROR_TARGET_PUBLIC_KEY="pilot.pem" # relative to cloud/keys directory.
   MIRROR_NUM_REPLICAS_PER_ZONE=2
   MONITORING="gcm"
   ```

1. Build & push Docker images:

   ```bash
   export PROJECT="my-project"
   make -j24
   sudo docker build -f Dockerfile-ct-mirror -t gcr.io/${PROJECT}/super_mirror:test .
   sudo docker build -t gcr.io/${PROJECT}/etcd:test cloud/etcd
   gcloud docker push gcr.io/${PROJECT}/super_mirror:test
   gcloud docker push gcr.io/${PROJECT}/etcd:test
   ```

   If using Prometheus, also run:

   ```bash
   sudo docker build -t gcr.io/${PROJECT}/prometheus:test cloud/prometheus`
   gcloud docker push gcr.io/${PROJECT}/prometheus:test
   ```

1. Start mirror:

   ```bash
   ./cloud/google/create_new_cluster.sh path/to/your/config
   ```

1. Configure monitoring dashboard and alerts
   * if using GCM click on `Monitoring > Dashboards & alerts` in your (cloud.google.com)[https://cloud.google.com]
     console.
   * if using Prometheus configure it on your Prometheus instance.
     You may find the following command useful while developing your Prometheus
     config:

     ```bash
     # Forward requests to localhost:9090 on your machine to a Prometheus
     # instance
     gcloud compute ssh <Prometheus host name> --ssh-flag="-L localhost:9090:localhost:9090"
     ```

