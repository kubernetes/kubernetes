# Jenkins

[Jenkins](http://jenkins-ci.org/) is a pluggable continuous
integration system. The Google team is running two Jenkins servers in GCE for
the Kubernetes project. The post-commit instance runs continuous builds, unit
tests, integration tests, code verification tests, and end-to-end tests on
multiple providers using the latest commits to the Kubernetes repo from the
master and release branches. The PR Jenkins instance runs these tests on each
PR by a trusted contributor, it but only runs a subset of the end-to-end tests
and only on GCE.

## General flow
The flow of the post-commit Jenkins instance:
* Under the `kubernetes-build` job: Every 2 minutes, Jenkins polls for a batch
  of new commits, after which it runs the `build.sh` script (in this directory)
  on the latest tip. This results in build assets getting pushed to GCS and the
  `latest.txt` file in the `ci` bucket being updated.
* On trigger, and every half hour (which effectively means all the time, unless
  we're failing cluster creation), e2e variants run, on the latest build assets
  in GCS:
  * `kubernetes-e2e-gce`: Standard GCE e2e.
  * `kubernetes-e2e-gke`: GKE provider e2e, with head k8s client and GKE
    creating clusters at its default version.
  * `kubernetes-e2e-aws`: AWS provider e2e. This only runs once a day.
* Each job will not run concurrently with itself, so, for instance,
  Jenkins executor will only ever run one `kubernetes-build`
  job. However, it may run the jobs in parallel,
  i.e. `kubernetes-build` may be run at the same time as
  `kubernetes-e2e-gce`. For this reason, you may see your changes
  pushed to our GCS bucket rapidly, but they may take some time to
  fully work through Jenkins. Or you may get lucky and catch the
  train in 5 minutes.
* There are many jobs not listed here, including upgrade tests, soak tests, and
  tests for previous releases.

## Scripts

The scripts in this directory are directly used by Jenkins, either by
curl from githubusercontent (if we don't have a git checkout handy) or
by executing it from the git checkout. Since Jenkins is an entity
outside this repository, it's tricky to keep documentation for it up
to date quickly. However, the scripts themselves attempt to provide
color for the configuration(s) that each script runs in.

## GCS Log Format

Our `upload-to-gcs.sh` script runs at the start and end of every job. Logs on
post-commit Jenkins go under `gs://kubernetes-jenkins/logs/`. Logs on PR
Jenkins go under `gs://kubernetes-jenkins-pull/pr-logs/pull/PULL_NUMBER/`.
Individual run logs go into the `JOB_NAME/BUILD_NUMBER` folder.

At the start of the job, it uploads `started.json` containing the version of
Kubernetes under test and the timestamp.

At the end, it uploads `finished.json` containing the result and timestamp, as
well as the build log into `build-log.txt`. Under `artifacts/` we put our
test results in `junit_XY.xml`, along with gcp resource lists and cluster logs.

It also updates `latest-build.txt` at the end to point to this build number.
In the end, the directory structure looks like this:

```
gs://kubernetes-jenkins/logs/kubernetes-e2e-gce/
  latest-build.txt
  12345/
    build-log.txt
    started.json
    finished.json
    artifacts/
      gcp-resources-{before, after}.txt
      junit_{00, 01, ...}.xml
      jenkins-e2e-master/{kube-apiserver.log, ...}
      jenkins-e2e-node-abcd/{kubelet.log, ...}
  12344/
    ...
```

The munger uses `latest-build.txt` and the JUnit reports to figure out whether
or not the job is healthy.

## Job Builder

New jobs should be specified as YAML files to be processed by [Jenkins Job
Builder](http://docs.openstack.org/infra/jenkins-job-builder/). The YAML files
live in `jenkins/job-configs` and its subfolders **in the
[kubernetes/test-infra repository](https://github.com/kubernetes/test-infra)**.
Jenkins runs Jenkins Job Builder in a Docker container defined in
`job-builder-image`, and triggers it using `update-jobs.sh`. Jenkins Job Builder
uses a config file called
[jenkins_jobs.ini](http://docs.openstack.org/infra/jenkins-job-builder/execution.html)
which contains the location and credentials of the Jenkins server.

E2E Job definitions are templated to avoid code duplication. To add a new job,
add a new entry to the appropriate `project`.
[This](https://github.com/kubernetes/kubernetes/commit/eb273e5a4bdd3905f881563ada4e6543c7eb96b5)
is an example of a commit which does this. If necessary, create a new project, as in
[this](https://github.com/kubernetes/kubernetes/commit/09c27cdabc300e0420a2914100bedb565c23ed73)
commit.

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/hack/jenkins/README.md?pixel)]()
