# Cloud Bigtable on Managed VMs using Go
# (Hello World for Cloud Bigtable)

This app counts how often each user visits.

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
1. Download App Engine SDK for Go.
  1. `go get -u google.golang.org/appengine/...`
1. In helloworld.go, change the constants `project`, `zone` and `cluster`

## Running locally

1. From the sample project folder, `gcloud preview app run app.yaml`

## Deploying on Google App Engine Managed VM

1. Install and start [Docker](https://cloud.google.com/appengine/docs/managed-vms/getting-started#install_docker).
1. From the sample project folder, `aedeploy gcloud preview app deploy app.yaml`
