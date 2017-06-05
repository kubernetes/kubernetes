# Cloud Bigtable Hello World in Go

This is a simple application that demonstrates using the [Google Cloud APIs Go
Client Library](https://github.com/GoogleCloudPlatform/google-cloud-go) to connect
to and interact with Cloud Bigtable.

## Prerequisites

1. Set up Cloud Console.
  1. Go to the [Cloud Console](https://cloud.google.com/console) and create or select your project.
     You will need the project ID later.
  1. Go to **Settings > Project Billing Settings** and enable billing.
  1. Select **APIs & Auth > APIs**.
  1. Enable the **Cloud Bigtable API** and the **Cloud Bigtable Admin API**.
  (You may need to search for the API).
1. Set up gcloud.
  1. `gcloud components update`
  1. `gcloud auth login`
  1. `gcloud config set project PROJECT_ID`
1. Provision a Cloud Bigtable instance
  1. Follow the instructions in the [user
documentation](https://cloud.google.com/bigtable/docs/creating-instance) to
create a Google Cloud Platform project and Cloud Bigtable instance if necessary.
  1. You'll need to reference your project id and instance id to run the application.

## Running

1. From the hello_world example folder, `go run main.go -project PROJECT_ID -instance INSTANCE_ID`, substituting your project id and instance id.

## Cleaning up

To avoid incurring extra charges to your Google Cloud Platform account, remove
the resources created for this sample.

1.  Go to the Clusters page in the [Cloud
    Console](https://console.cloud.google.com).

    [Go to the Clusters page](https://console.cloud.google.com/project/_/bigtable/clusters)

1.  Click the cluster name.

1.  Click **Delete**.

    ![Delete](https://cloud.google.com/bigtable/img/delete-quickstart-cluster.png)

1. Type the cluster ID, then click **Delete** to delete the cluster.
