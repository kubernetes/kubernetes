# Jenkins

[Jenkins](http://jenkins-ci.org/) is a pluggable continuous
integration system. The Google team is running a Jenkins server on a
private GCE instance for the Kubernetes project in order to run longer
integration tests, continuously, on different providers. Currently, we
(Google) are only running Jenkins on our own providers (GCE and GKE)
in different flavors.

## General flow
The flow of the Google Jenkins server:
* Under the `kubernetes-build` job: Every 5 minutes, Jenkins polls for a batch of new commits, after which it runs the `build.sh` script (in this directory) on the latest tip. This results in build assets getting pushed to GCS and the `latest.txt` file in the `ci` bucket being updated. That job then triggers `kubernetes-e2e-*`.
* On trigger, and every half hour (which effectively means all the time, unless we're failing cluster creation), e2e variants run, on the latest build assets in GCS:
  * `kubernetes-e2e-gce`: Standard GCE e2e
  * `kubernetes-e2e-gke`: GKE provider e2e, with head k8s client and GKE creating clusters at its default version
  * `kubernetes-e2e-gke-ci`: GKE provider e2e, with head k8s client and GKE creating clusters at the head k8s version
* Each job will not run concurrently with itself, so, for instance,
  Jenkins executor will only ever run one `kubernetes-build`
  job. However, it may run the jobs in parallel,
  i.e. `kubernetes-build` may be run at the same time as
  `kubernetes-e2e-gce`. For this reason, you may see your changes
  pushed to our GCS bucket rapidly, but they may take some time to
  fully work through Jenkins. Or you may get lucky and catch the
  train in 5 minutes.

## Scripts

The scripts in this directory are directly used by Jenkins, either by
curl from githubusercontent (if we don't have a git checkout handy) or
by executing it from the git checkout. Since Jenkins is an entity
outside this repository, it's tricky to keep documentation for it up
to date quickly. However, the scripts themselves attempt to provide
color for the configuration(s) that each script runs in.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/hack/jenkins/README.md?pixel)]()
