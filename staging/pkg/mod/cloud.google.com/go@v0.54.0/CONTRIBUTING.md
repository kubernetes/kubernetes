# Contributing

1. [File an issue](https://github.com/googleapis/google-cloud-go/issues/new/choose).
   The issue will be used to discuss the bug or feature and should be created
   before sending a CL.

1. [Install Go](https://golang.org/dl/).
    1. Ensure that your `GOBIN` directory (by default `$(go env GOPATH)/bin`)
    is in your `PATH`.
    1. Check it's working by running `go version`.
        * If it doesn't work, check the install location, usually
        `/usr/local/go`, is on your `PATH`.

1. Sign one of the
[contributor license agreements](#contributor-license-agreements) below.

1. Run `go get golang.org/x/review/git-codereview && go install golang.org/x/review/git-codereview`
to install the code reviewing tool.

    1. Ensure it's working by running `git codereview` (check your `PATH` if
    not).

    1. If you would like, you may want to set up aliases for `git-codereview`,
    such that `git codereview change` becomes `git change`. See the
    [godoc](https://pkg.go.dev/golang.org/x/review/git-codereview) for details.

        * Should you run into issues with the `git-codereview` tool, please note
        that all error messages will assume that you have set up these aliases.

1. Change to a directory of your choosing and clone the repo.

    ```
    cd ~/code
    git clone https://code.googlesource.com/gocloud
    ```

    * If you have already checked out the source, make sure that the remote
    `git` `origin` is https://code.googlesource.com/gocloud:

        ```
        git remote -v
        # ...
        git remote set-url origin https://code.googlesource.com/gocloud
        ```

    * The project uses [Go Modules](https://blog.golang.org/using-go-modules)
    for dependency management See
    [`gopls`](https://github.com/golang/go/wiki/gopls) for making your editor
    work with modules.

1. Change to the project directory:

    ```
    cd ~/code/gocloud
    ```

1. Make sure your `git` auth is configured correctly by visiting
https://code.googlesource.com, clicking "Generate Password" at the top-right,
and following the directions. Otherwise, `git codereview mail` in the next step
will fail.

1. Now you are ready to make changes. Don't create a new branch or make commits in the traditional
way. Use the following`git codereview` commands to create a commit and create a Gerrit CL:

    ```
    git codereview change <branch-name> # Use this instead of git checkout -b <branch-name>
    # Make changes.
    git add ...
    git codereview change # Use this instead of git commit
    git codereview mail # If this fails, the error message will contain instructions to fix it.
    ```

    * This will create a new `git` branch for you to develop on. Once your
    change is merged, you can delete this branch.

1. As you make changes for code review, ammend the commit and re-mail the
change:

    ```
    # Make more changes.
    git add ...
    git codereview change
    git codereview mail
    ```

    * **Warning**: do not change the `Change-Id` at the bottom of the commit
    message - it's how Gerrit knows which change this is (or if it's new).

    * When you fixes issues from code review, respond to each code review
    message then click **Reply** at the top of the page.

    * Each new mailed amendment will create a new patch set for
    your change in Gerrit. Patch sets can be compared and reviewed.

    * **Note**: if your change includes a breaking change, our breaking change
    detector will cause CI/CD to fail. If your breaking change is acceptable
    in some way, add a `BREAKING_CHANGE_ACCEPTABLE=<reason>` line to the commit
    message to cause the detector not to be run and to make it clear why that is
    acceptable.

1. Finally, add reviewers to your CL when it's ready for review. Reviewers will
not be added automatically. If you're not sure who to add for your code review,
add tbp@, cbro@, and codyoss@.


## Integration Tests

In addition to the unit tests, you may run the integration test suite. These
directions describe setting up your environment to run integration tests for
_all_ packages: note that many of these instructions may be redundant if you
intend only to run integration tests on a single package.

#### GCP Setup

To run the integrations tests, creation and configuration of two projects in
the Google Developers Console is required: one specifically for Firestore
integration tests, and another for all other integration tests. We'll refer to
these projects as "general project" and "Firestore project".

After creating each project, you must [create a service account](https://developers.google.com/identity/protocols/OAuth2ServiceAccount#creatinganaccount)
for each project. Ensure the project-level **Owner**
[IAM role](console.cloud.google.com/iam-admin/iam/project) role is added to
each service account. During the creation of the service account, you should
download the JSON credential file for use later.

Next, ensure the following APIs are enabled in the general project:

- BigQuery API
- BigQuery Data Transfer API
- Cloud Dataproc API
- Cloud Dataproc Control API Private
- Cloud Datastore API
- Cloud Firestore API
- Cloud Key Management Service (KMS) API
- Cloud Natural Language API
- Cloud OS Login API
- Cloud Pub/Sub API
- Cloud Resource Manager API
- Cloud Spanner API
- Cloud Speech API
- Cloud Translation API
- Cloud Video Intelligence API
- Cloud Vision API
- Compute Engine API
- Compute Engine Instance Group Manager API
- Container Registry API
- Firebase Rules API
- Google Cloud APIs
- Google Cloud Deployment Manager V2 API
- Google Cloud SQL
- Google Cloud Storage
- Google Cloud Storage JSON API
- Google Compute Engine Instance Group Updater API
- Google Compute Engine Instance Groups API
- Kubernetes Engine API
- Stackdriver Error Reporting API

Next, create a Datastore database in the general project, and a Firestore
database in the Firestore project.

Finally, in the general project, create an API key for the translate API:

- Go to GCP Developer Console.
- Navigate to APIs & Services > Credentials.
- Click Create Credentials > API Key.
- Save this key for use in `GCLOUD_TESTS_API_KEY` as described below.

#### Local Setup

Once the two projects are created and configured, set the following environment
variables:

- `GCLOUD_TESTS_GOLANG_PROJECT_ID`: Developers Console project's ID (e.g.
bamboo-shift-455) for the general project.
- `GCLOUD_TESTS_GOLANG_KEY`: The path to the JSON key file of the general
project's service account.
- `GCLOUD_TESTS_GOLANG_FIRESTORE_PROJECT_ID`: Developers Console project's ID
(e.g. doorway-cliff-677) for the Firestore project.
- `GCLOUD_TESTS_GOLANG_FIRESTORE_KEY`: The path to the JSON key file of the
Firestore project's service account.
- `GCLOUD_TESTS_GOLANG_KEYRING`: The full name of the keyring for the tests,
in the form
"projects/P/locations/L/keyRings/R". The creation of this is described below.
- `GCLOUD_TESTS_API_KEY`: API key for using the Translate API.
- `GCLOUD_TESTS_GOLANG_ZONE`: Compute Engine zone.

Install the [gcloud command-line tool][gcloudcli] to your machine and use it to
create some resources used in integration tests.

From the project's root directory:

``` sh
# Sets the default project in your env.
$ gcloud config set project $GCLOUD_TESTS_GOLANG_PROJECT_ID

# Authenticates the gcloud tool with your account.
$ gcloud auth login

# Create the indexes used in the datastore integration tests.
$ gcloud datastore indexes create datastore/testdata/index.yaml

# Creates a Google Cloud storage bucket with the same name as your test project,
# and with the Stackdriver Logging service account as owner, for the sink
# integration tests in logging.
$ gsutil mb gs://$GCLOUD_TESTS_GOLANG_PROJECT_ID
$ gsutil acl ch -g cloud-logs@google.com:O gs://$GCLOUD_TESTS_GOLANG_PROJECT_ID

# Creates a PubSub topic for integration tests of storage notifications.
$ gcloud beta pubsub topics create go-storage-notification-test
# Next, go to the Pub/Sub dashboard in GCP console. Authorize the user
# "service-<numberic project id>@gs-project-accounts.iam.gserviceaccount.com"
# as a publisher to that topic.

# Creates a Spanner instance for the spanner integration tests.
$ gcloud beta spanner instances create go-integration-test --config regional-us-central1 --nodes 10 --description 'Instance for go client test'
# NOTE: Spanner instances are priced by the node-hour, so you may want to
# delete the instance after testing with 'gcloud beta spanner instances delete'.

$ export MY_KEYRING=some-keyring-name
$ export MY_LOCATION=global
# Creates a KMS keyring, in the same location as the default location for your
# project's buckets.
$ gcloud kms keyrings create $MY_KEYRING --location $MY_LOCATION
# Creates two keys in the keyring, named key1 and key2.
$ gcloud kms keys create key1 --keyring $MY_KEYRING --location $MY_LOCATION --purpose encryption
$ gcloud kms keys create key2 --keyring $MY_KEYRING --location $MY_LOCATION --purpose encryption
# Sets the GCLOUD_TESTS_GOLANG_KEYRING environment variable.
$ export GCLOUD_TESTS_GOLANG_KEYRING=projects/$GCLOUD_TESTS_GOLANG_PROJECT_ID/locations/$MY_LOCATION/keyRings/$MY_KEYRING
# Authorizes Google Cloud Storage to encrypt and decrypt using key1.
gsutil kms authorize -p $GCLOUD_TESTS_GOLANG_PROJECT_ID -k $GCLOUD_TESTS_GOLANG_KEYRING/cryptoKeys/key1
```

#### Running

Once you've done the necessary setup, you can run the integration tests by
running:

``` sh
$ go test -v cloud.google.com/go/...
```

#### Replay

Some packages can record the RPCs during integration tests to a file for
subsequent replay. To record, pass the `-record` flag to `go test`. The
recording will be saved to the _package_`.replay` file. To replay integration
tests from a saved recording, the replay file must be present, the `-short`
flag must be passed to `go test`, and the `GCLOUD_TESTS_GOLANG_ENABLE_REPLAY`
environment variable must have a non-empty value.

## Contributor License Agreements

Before we can accept your pull requests you'll need to sign a Contributor
License Agreement (CLA):

- **If you are an individual writing original source code** and **you own the
intellectual property**, then you'll need to sign an [individual CLA][indvcla].
- **If you work for a company that wants to allow you to contribute your
work**, then you'll need to sign a [corporate CLA][corpcla].

You can sign these electronically (just scroll to the bottom). After that,
we'll be able to accept your pull requests.

## Contributor Code of Conduct

As contributors and maintainers of this project,
and in the interest of fostering an open and welcoming community,
we pledge to respect all people who contribute through reporting issues,
posting feature requests, updating documentation,
submitting pull requests or patches, and other activities.

We are committed to making participation in this project
a harassment-free experience for everyone,
regardless of level of experience, gender, gender identity and expression,
sexual orientation, disability, personal appearance,
body size, race, ethnicity, age, religion, or nationality.

Examples of unacceptable behavior by participants include:

* The use of sexualized language or imagery
* Personal attacks
* Trolling or insulting/derogatory comments
* Public or private harassment
* Publishing other's private information,
such as physical or electronic
addresses, without explicit permission
* Other unethical or unprofessional conduct.

Project maintainers have the right and responsibility to remove, edit, or reject
comments, commits, code, wiki edits, issues, and other contributions
that are not aligned to this Code of Conduct.
By adopting this Code of Conduct,
project maintainers commit themselves to fairly and consistently
applying these principles to every aspect of managing this project.
Project maintainers who do not follow or enforce the Code of Conduct
may be permanently removed from the project team.

This code of conduct applies both within project spaces and in public spaces
when an individual is representing the project or its community.

Instances of abusive, harassing, or otherwise unacceptable behavior
may be reported by opening an issue
or contacting one or more of the project maintainers.

This Code of Conduct is adapted from the [Contributor Covenant](http://contributor-covenant.org), version 1.2.0,
available at [http://contributor-covenant.org/version/1/2/0/](http://contributor-covenant.org/version/1/2/0/)

[gcloudcli]: https://developers.google.com/cloud/sdk/gcloud/
[indvcla]: https://developers.google.com/open-source/cla/individual
[corpcla]: https://developers.google.com/open-source/cla/corporate
