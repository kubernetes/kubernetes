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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Externally accessible Kubernetes test results

## Objective

Make kubernetes test results accessible to anyone on the internet

## Background

For the time being Google runs most of the testing for kubernetes. As a result,
developers working inside Google have access to better test result information
than everyone else. We want to change that. We want the best tools for looking
at test results inside and outside Google to be one and the same.

Specifically we have two primary ways of looking at test results:

1. A summary from the past twenty-four hours
2. A view of recent test results updated every thirty minutes

Option 1 is now available to everyone at
http://storage.googleapis.com/kubernetes-test-history/static/index.html

Option 2 is available to people inside of Google, but we will make it available
to everyone in time for the release of 1.3

![test result view](../images/test-results.png)

## Detailed design

We will take the tool we have access to internally and put it on a public
website. Then we will update the tool to collect information from multiple
sources.

### Storing test result information

* We will associate each jenkins jobs with a tab name
* We will associate each jenkins job with a GCS prefix
  * Example: gs://kubernetes-jenkins/logs/kubernetes-e2e-gce/
  * The owner of this bucket will grant a specific service account read access to
    the bucket as well as files under this prefix
* Each run of the job will put files in a subfolder
  * Example gs://kubernetes-jenkins/logs/kubernetes-e2e-gce/15039/
  * Subfolders will have monotonically increasing numbers.
  * This means that the next run id will be >= 15040
* Each run will have the following files:
  * Started.json
      * Version string and timestamp (seconds since epoch) keys
  * Finished.json
      * Overall pass/fail result and timestamp keys
  * artifacts/junit\_NN.xml where 01 <= NN <= 99
      * Following standard junit.xml format outputted by jenkins

### Collecitng test result information

* Google will run a service that collects information about the list of jobs
  * It will update this information every 30 minutes
* It will then find the GCS prefix associated with that tab
* It will then attempt to list objects under that prefix using the specified
* service account
* It will download the well known files under those prefixes:
  * (started.json, finished.json and artifacts/junit\_NN.xml)
  * Initially every object each time it runs
* This information will allow it to maintain information about the state of test
  results
  * Jobs will have a list of runs sorted by id
  * Each run will have
    * An id
    * A start time
    * A duration
    * A completion status (running, completed, failed)
    * A list of test cases
  * Each test case will have
    * A name of the test case
    * A start time
    * A duration
    * A completion status (passed, failed)


#### Future optimization ideas

* Only list new runs
* Only download new and/or updated objects.
* Download as fast as possible
* Device a service for pushing results rather than polling.
* Lots of opportunities to collect different/better information:
  * example: failure stack trace

### Displaying test result information

* Google will run a service that displays the state of collected test results
  * Available to all kubernetes developers
* The service will display a list of tabs on the top of the screen
  * Each tab is associated with a jenkins job
* Each tab will display one column per run
  * The column will note the run id and start time
* Each tab will display one row per test case
  * The test case header will display the name of the test case
* Each cell will display the test case result during that run
  * Green = pass, red == fail, missing == run did not run this test case
  * Clicking the link will display logs collected in GCS for that run.


#### Visual customiation

The user can control a variety of information:

* The width of columns
* The sort order of the test cases
* Test case filter/grouping
* Graphing the runtime of each test case


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/test-results.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
