<!--[metadata]>
+++
title = "Installation on Google Cloud Platform"
description = "Installation instructions for Docker on the Google Cloud Platform."
keywords = ["Docker, Docker documentation, installation, google, Google Compute Engine,  Google Cloud Platform"]
[menu.main]
parent = "smn_cloud"
+++
<![end-metadata]-->

# Google Cloud Platform

## QuickStart with Container-optimized Google Compute Engine images

1. Go to [Google Cloud Console][1] and create a new Cloud Project with
   [Compute Engine enabled][2]

2. Download and configure the [Google Cloud SDK][3] to use your
   project with the following commands:

        $ curl -sSL https://sdk.cloud.google.com | bash
        $ gcloud auth login
        $ gcloud config set project <google-cloud-project-id>

3. Start a new instance using the latest [Container-optimized image][4]:
   (select a zone close to you and the desired instance size)

        $ gcloud compute instances create docker-playground \
          --image container-vm \
          --zone us-central1-a \
          --machine-type f1-micro

4. Connect to the instance using SSH:

        $ gcloud compute ssh --zone us-central1-a docker-playground
        docker-playground:~$ sudo docker run hello-world
	Hello from Docker.
	This message shows that your installation appears to be working correctly.
	...

Read more about [deploying Containers on Google Cloud Platform][5].

[1]: https://cloud.google.com/console
[2]: https://developers.google.com/compute/docs/signup
[3]: https://developers.google.com/cloud/sdk
[4]: https://developers.google.com/compute/docs/containers#container-optimized_google_compute_engine_images
[5]: https://developers.google.com/compute/docs/containers
