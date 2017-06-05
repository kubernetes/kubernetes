# Contributing

1. Sign one of the contributor license agreements below.
1. `go get golang.org/x/review/git-codereview` to install the code reviewing tool.
1. Get the cloud package by running `go get -d cloud.google.com/go`.
    1. If you have already checked out the source, make sure that the remote git
       origin is https://code.googlesource.com/gocloud:

            git remote set-url origin https://code.googlesource.com/gocloud
1. Make sure your auth is configured correctly by visiting
   https://code.googlesource.com, clicking "Generate Password", and following
   the directions.
1. Make changes and create a change by running `git codereview change <name>`,
provide a commit message, and use `git codereview mail` to create a Gerrit CL.
1. Keep amending to the change and mail as your receive feedback.

## Integration Tests

In addition to the unit tests, you may run the integration test suite.

To run the integrations tests, creating and configuration of a project in the
Google Developers Console is required.

After creating a project, you must [create a service account](https://developers.google.com/identity/protocols/OAuth2ServiceAccount#creatinganaccount).
Ensure the project-level **Owner** [IAM role](console.cloud.google.com/iam-admin/iam/project)
(or **Editor** and **Logs Configuration Writer** roles) are added to the
service account.

Once you create a project, set the following environment variables to be able to
run the against the actual APIs.

- **GCLOUD_TESTS_GOLANG_PROJECT_ID**: Developers Console project's ID (e.g. bamboo-shift-455)
- **GCLOUD_TESTS_GOLANG_KEY**: The path to the JSON key file.

Install the [gcloud command-line tool][gcloudcli] to your machine and use it
to create the indexes used in the datastore integration tests with indexes
found in `datastore/testdata/index.yaml`:

From the project's root directory:

``` sh
# Set the default project in your env
$ gcloud config set project $GCLOUD_TESTS_GOLANG_PROJECT_ID

# Authenticate the gcloud tool with your account
$ gcloud auth login

# Create the indexes
$ gcloud preview datastore create-indexes datastore/testdata/index.yaml
```

The Sink integration tests in preview/logging require a Google Cloud storage
bucket with the same name as your test project, and with the Stackdriver Logging
service account as owner:
``` sh
$ gsutil mb gs://$GCLOUD_TESTS_GOLANG_PROJECT_ID
$ gsutil acl ch -g cloud-logs@google.com:O gs://$GCLOUD_TESTS_GOLANG_PROJECT_ID
```

Once you've set the environment variables, you can run the integration tests by
running:

``` sh
$ go test -v cloud.google.com/go/...
```

## Contributor License Agreements

Before we can accept your pull requests you'll need to sign a Contributor
License Agreement (CLA):

- **If you are an individual writing original source code** and **you own the
- intellectual property**, then you'll need to sign an [individual CLA][indvcla].
- **If you work for a company that wants to allow you to contribute your work**,
then you'll need to sign a [corporate CLA][corpcla].

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
