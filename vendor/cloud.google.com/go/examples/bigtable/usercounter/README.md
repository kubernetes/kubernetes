# User Counter
# (Cloud Bigtable on Managed VMs using Go)

This app counts how often each user visits. The app uses Cloud Bigtable to store the visit counts for each user.

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
1. In main.go, change the `project` and `instance` constants.

## Running locally

1. From the sample project folder, `dev_appserver.py app.yaml`.

## Deploying on Google App Engine flexible environment

Follow the [deployment instructions](https://cloud.google.com/appengine/docs/flexible/go/testing-and-deploying-your-app).
