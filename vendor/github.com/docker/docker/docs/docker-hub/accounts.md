<!--[metadata]>
+++
title = "Accounts on Docker Hub"
description = "Docker Hub accounts"
keywords = ["Docker, docker, registry, accounts, plans, Dockerfile, Docker Hub, docs,  documentation"]
[menu.main]
parent = "smn_pubhub"
weight = 1
+++
<![end-metadata]-->

# Accounts on Docker Hub

## Docker Hub accounts

You can `search` for Docker images and `pull` them from [Docker
Hub](https://hub.docker.com) without signing in or even having an
account. However, in order to `push` images, leave comments or to *star*
a repository, you are going to need a [Docker
Hub](https://hub.docker.com) account.

### Registration for a Docker Hub account

You can get a [Docker Hub](https://hub.docker.com) account by
[signing up for one here](https://hub.docker.com/account/signup/). A valid
email address is required to register, which you will need to verify for
account activation.

### Email activation process

You need to have at least one verified email address to be able to use your
[Docker Hub](https://hub.docker.com) account. If you can't find the validation email,
you can request another by visiting the [Resend Email Confirmation](
https://hub.docker.com/account/resend-email-confirmation/) page.

### Password reset process

If you can't access your account for some reason, you can reset your password
from the [*Password Reset*](https://hub.docker.com/account/forgot-password/)
page.

## Organizations and groups

A Docker Hub organization contains public and private repositories just like
a user account. Access to push, pull or create these organisation owned repositories
is allocated by defining groups of users and then assigning group rights to
specific repositories. This allows you to distribute limited access
Docker images, and to select which Docker Hub users can publish new images.

### Creating and viewing organizations

You can see what organizations [you belong to and add new organizations](
https://hub.docker.com/account/organizations/) from the Account Settings
tab. They are also listed below your user name on your repositories page
and in your account profile.

![organizations](/docker-hub/hub-images/orgs.png)

### Organization groups

Users in the `Owners` group of an organization can create and modify the
membership of groups.

Unless they are the organization's `Owner`, users can only see groups of which they
are members.

![groups](/docker-hub/hub-images/groups.png)

### Repository group permissions

Use organization groups to manage the users that can interact with your repositories.

You must be in an organization's `Owners` group to create a new group, Hub
repository, or automated build. As an `Owner`, you then delegate the following
repository access rights to groups:

| Access Right | Description                                                                                                                                                                                                                                                                |
|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Read`       | Users with this right can view, search, and pull a private repository.                                                                                                                                                                                                     |
| `Write`      | Users with this right can push to non-automated repositories on the Docker Hub.                                                                                                                                                                                            |
| `Admin`      | Users with this right can modify a repository's "Description", "Collaborators" rights. They can also mark a repository as unlisted, change its  "Public/Private" status and "Delete" the repository. Finally, `Admin` rights are required to read the build log on a repo. |
|              |                                                                                                                                                                                                                                                                            |

Regardless of their actual access rights, users with unverified email addresses
have `Read` access to the repository. Once they have verified their address,
they have their full access rights as granted on the organization.
