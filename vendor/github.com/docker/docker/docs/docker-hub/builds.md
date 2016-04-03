<!--[metadata]>
+++
title = "Automated Builds on Docker Hub"
description = "Docker Hub Automated Builds"
keywords = ["Docker, docker, registry, accounts, plans, Dockerfile, Docker Hub, docs, documentation, trusted, builds, trusted builds,  automated builds"]
[menu.main]
parent = "smn_pubhub"
weight = 3
+++
<![end-metadata]-->

# Automated Builds on Docker Hub

## About Automated Builds

*Automated Builds* are a special feature of Docker Hub which allow you to
use [Docker Hub's](https://hub.docker.com) build clusters to automatically
create images from a GitHub or Bitbucket repository containing a `Dockerfile`
The system will clone your repository and build the image described by the
`Dockerfile` using the directory the `Dockerfile` is in (and subdirectories)
as the build context. The resulting automated image will then be uploaded
to the Docker Hub registry and marked as an *Automated Build*.

Automated Builds have several advantages:

* Users of *your* Automated Build can trust that the resulting
image was built exactly as specified.
* The `Dockerfile` will be available to anyone with access to
your repository on the Docker Hub registry.
* Because the process is automated, Automated Builds help to
make sure that your repository is always up to date.

Automated Builds are supported for both public and private repositories
on both [GitHub](http://github.com) and [Bitbucket](https://bitbucket.org/).

To use Automated Builds, you must have an [account on Docker Hub](
https://docs.docker.com/userguide/dockerhub/#creating-a-docker-hub-account)
and on GitHub and/or Bitbucket. In either case, the account needs
to be properly validated and activated before you can link to it.

The first time you to set up an Automated Build, your
[Docker Hub](https://hub.docker.com) account will need to be linked to
a GitHub or Bitbucket account.
This will allow the registry to see your repositories.

If you have previously linked your Docker Hub account, and want to view or modify
that link, click on the "Manage - Settings" link in the sidebar, and then
"Linked Accounts" in your Settings sidebar.

## Automated Builds from GitHub

If you've previously linked your Docker Hub account to your GitHub account,
you'll be able to skip to the [Creating an Automated Build](#creating-an-automated-build).

### Linking your Docker Hub account to a GitHub account

> *Note:*
> Automated Builds currently require *read* and *write* access since
> [Docker Hub](https://hub.docker.com) needs to setup a GitHub service
> hook. We have no choice here, this is how GitHub manages permissions, sorry!
> We do guarantee nothing else will be touched in your account.

To get started, log into your Docker Hub account and click the
"+ Add Repository" button at the upper right of the screen. Then select
[Automated Build](https://registry.hub.docker.com/builds/add/).

Select the [GitHub service](https://registry.hub.docker.com/associate/github/).

When linking to GitHub, you'll need to select either "Public and Private",
or "Limited" linking.

The "Public and Private" option is the easiest to use,
as it grants the Docker Hub full access to all of your repositories. GitHub
also allows you to grant access to repositories belonging to your GitHub
organizations.

By choosing the "Limited" linking, your Docker Hub account only gets permission
to access your public data and public repositories.

Follow the onscreen instructions to authorize and link your
GitHub account to Docker Hub. Once it is linked, you'll be able to
choose a source repository from which to create the Automatic Build.

You will be able to review and revoke Docker Hub's access by visiting the
[GitHub User's Applications settings](https://github.com/settings/applications).

> **Note**: If you delete the GitHub account linkage that is used for one of your
> automated build repositories, the previously built images will still be available.
> If you re-link to that GitHub account later, the automated build can be started
> using the "Start Build" button on the Hub, or if the webhook on the GitHub repository
> still exists, will be triggered by any subsequent commits.

### Auto builds and limited linked GitHub accounts.

If you selected to link your GitHub account with only a "Limited" link, then
after creating your automated build, you will need to either manually trigger a
Docker Hub build using the "Start a Build" button, or add the GitHub webhook
manually, as described in [GitHub Service Hooks](#github-service-hooks).

### Changing the GitHub user link

If you want to remove, or change the level of linking between your GitHub account
and the Docker Hub, you need to do this in two places.

First, remove the "Linked Account" from your Docker Hub "Settings".
Then go to your GitHub account's Personal settings, and in the "Applications"
section, "Revoke access".

You can now re-link your account at any time.

### GitHub organizations

GitHub organizations and private repositories forked from organizations will be
made available to auto build using the "Docker Hub Registry" application, which
needs to be added to the organization - and then will apply to all users.

To check, or request access, go to your GitHub user's "Setting" page, select the
"Applications" section from the left side bar, then click the "View" button for
"Docker Hub Registry".

![Check User access to GitHub](/docker-hub/hub-images/gh-check-user-org-dh-app-access.png)

The organization's administrators may need to go to the Organization's "Third
party access" screen in "Settings" to Grant or Deny access to the Docker Hub
Registry application. This change will apply to all organization members.

![Check Docker Hub application access to Organization](/docker-hub/hub-images/gh-check-admin-org-dh-app-access.png)

More detailed access controls to specific users and GitHub repositories would be
managed using the GitHub People and Teams interfaces.

### Creating an Automated Build

You can [create an Automated Build](
https://registry.hub.docker.com/builds/github/select/) from any of your
public or private GitHub repositories that have a `Dockerfile`.

Once you've selected the source repository, you can then configure:

- The Hub user/org the repository is built to - either your Hub account name,
or the name of any Hub organizations your account is in
- The Docker repository name the image is built to
- If the Docker repository should be "Public" or "Private"
  You can change the accessibility options after the repository has been created.
  If you add a Private repository to a Hub user, then you can only add other users
  as collaborators, and those users will be able to view and pull all images in that 
  repository. To configure more granular access permissions, such as using groups of 
  users or allow different users access to different image tags, then you need
  to add the Private repository to a Hub organization that your user has Administrator
  privilege on.
- If you want the GitHub to notify the Docker Hub when a commit is made, and thus trigger
  a rebuild of all the images in this automated build.

You can also select one or more
- The git branch/tag, which repository sub-directory to use as the context
- The Docker image tag name

You can set a description for the repository by clicking "Description" link in the righthand side bar after the automated build - note that the "Full Description" will be over-written next build from the README.md file.
has been created.

### GitHub private submodules

If your GitHub repository contains links to private submodules, you'll get an
error message in your build.

Normally, the Docker Hub sets up a deploy key in your GitHub repository.
Unfortunately, GitHub only allows a repository deploy key to access a single repository.

To work around this, you need to create a dedicated user account in GitHub and attach
the automated build's deploy key that account. This dedicated build account
can be limited to read-only access to just the repositories required to build.

<table class="table table-bordered">
  <thead>
    <tr>
      <th>Step</th>
      <th>Screenshot</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.</td>
      <td><img src="/docker-hub/hub-images/gh_org_members.png"></td>
      <td>First, create the new account in GitHub. It should be given read-only
      access to the main repository and all submodules that are needed.</td>
    </tr>
    <tr>
      <td>2.</td>
      <td><img src="/docker-hub/hub-images/gh_team_members.png"></td>
      <td>This can be accomplished by adding the account to a read-only team in
      the organization(s) where the main GitHub repository and all submodule
      repositories are kept.</td>
    </tr>
    <tr>
      <td>3.</td>
      <td><img src="/docker-hub/hub-images/gh_repo_deploy_key.png"></td>
      <td>Next, remove the deploy key from the main GitHub repository. This can be done in the GitHub repository's "Deploy keys" Settings section.</td>
    </tr>
    <tr>
      <td>4.</td>
      <td><img src="/docker-hub/hub-images/deploy_key.png"></td>
      <td>Your automated build's deploy key is in the "Build Details" menu
      under "Deploy keys".</td>
    </tr>
    <tr>
      <td>5.</td>
      <td><img src="/docker-hub/hub-images/gh_add_ssh_user_key.png"></td>
      <td>In your dedicated GitHub User account, add the deploy key from your
      Docker Hub Automated Build.</td>
    </tr>
  </tbody>
</table>

### GitHub service hooks

The GitHub Service hook allows GitHub to notify the Docker Hub when something has
been committed to that git repository. You will need to add the Service Hook manually
if your GitHub account is "Limited" linked to the Docker Hub.

Follow the steps below to configure the GitHub Service hooks for your Automated Build:

<table class="table table-bordered">
  <thead>
    <tr>
      <th>Step</th>
      <th>Screenshot</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.</td>
      <td><img src="/docker-hub/hub-images/gh_settings.png"></td>
      <td>Log in to GitHub.com, and go to your Repository page. Click on "Settings" on
      the right side of the page. You must have admin privileges to the repository in order to do this.</td>
    </tr>
    <tr>
      <td>2.</td>
      <td><img src="/docker-hub/hub-images/gh_menu.png" alt="Webhooks & Services"></td>
      <td>Click on "Webhooks & Services" on the left side of the page.</td></tr>
      <tr><td>3.</td>
      <td><img src="/docker-hub/hub-images/gh_service_hook.png" alt="Find the service labeled Docker"></td>
      <td>Find the service labeled "Docker" (or click on "Add service") and click on it.</td></tr>
      <tr><td>4.</td>
      <td><img src="/docker-hub/hub-images/gh_docker-service.png" alt="Activate Service Hooks"></td>
      <td>Make sure the "Active" checkbox is selected and click the "Update service" button to save your changes.</td>
    </tr>
  </tbody>
</table>

## Automated Builds with Bitbucket

In order to setup an Automated Build, you need to first link your
[Docker Hub](https://hub.docker.com) account with a Bitbucket account.
This will allow the registry to see your repositories.

To get started, log into your Docker Hub account and click the
"+ Add Repository" button at the upper right of the screen. Then
select [Automated Build](https://registry.hub.docker.com/builds/add/).

Select the [Bitbucket source](
https://registry.hub.docker.com/associate/bitbucket/).

Then follow the onscreen instructions to authorize and link your
Bitbucket account to Docker Hub. Once it is linked, you'll be able
to choose a repository from which to create the Automatic Build.

### Creating an Automated Build

You can [create an Automated Build](
https://registry.hub.docker.com/builds/bitbucket/select/) from any of your
public or private Bitbucket repositories with a `Dockerfile`.

### Adding a Hook

When you link your Docker Hub account, a `POST` hook should get automatically
added to your Bitbucket repository. Follow the steps below to confirm or modify the
Bitbucket hooks for your Automated Build:

<table class="table table-bordered">
  <thead>
    <tr>
      <th>Step</th>
      <th>Screenshot</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1.</td>
      <td><img src="/docker-hub/hub-images/bb_menu.png" alt="Settings" width="180"></td>
      <td>Log in to Bitbucket.org and go to your Repository page. Click on "Settings" on
      the far left side of the page, under "Navigation". You must have admin privileges
      to the repository in order to do this.</td>
    </tr>
    <tr>
      <td>2.</td>
      <td><img src="/docker-hub/hub-images/bb_hooks.png" alt="Hooks" width="180"></td>
      <td>Click on "Hooks" on the near left side of the page, under "Settings".</td></tr>
    <tr>
      <td>3.</td>
      <td><img src="/docker-hub/hub-images/bb_post-hook.png" alt="Docker Post Hook"></td><td>You should now see a list of hooks associated with the repo, including a <code>POST</code> hook that points at
      registry.hub.docker.com/hooks/bitbucket.</td>
    </tr>
  </tbody>
</table>


## The Dockerfile and Automated Builds

During the build process, Docker will copy the contents of your `Dockerfile`.
It will also add it to the [Docker Hub](https://hub.docker.com) for the Docker
community (for public repositories) or approved team members/orgs (for private
repositories) to see on the repository page.

### README.md

If you have a `README.md` file in your repository, it will be used as the
repository's full description.The build process will look for a
`README.md` in the same directory as your `Dockerfile`.

> **Warning:**
> If you change the full description after a build, it will be
> rewritten the next time the Automated Build has been built. To make changes,
> modify the `README.md` from the Git repository.

## Remote Build triggers

If you need a way to trigger Automated Builds outside of GitHub or Bitbucket,
you can set up a build trigger. When you turn on the build trigger for an
Automated Build, it will give you a URL to which you can send POST requests.
This will trigger the Automated Build, much as with a GitHub webhook.

Build triggers are available under the Settings menu of each Automated Build
repository on the Docker Hub.

![Build trigger screen](/docker-hub/hub-images/build-trigger.png)

You can use `curl` to trigger a build:

```
$ curl --data "build=true" -X POST https://registry.hub.docker.com/u/svendowideit/testhook/trigger/be579c
82-7c0e-11e4-81c4-0242ac110020/
OK
```

> **Note:**
> You can only trigger one build at a time and no more than one
> every five minutes. If you already have a build pending, or if you
> recently submitted a build request, those requests *will be ignored*.
> To verify everything is working correctly, check the logs of last
> ten triggers on the settings page .

## Webhooks

Automated Builds also include a Webhooks feature. Webhooks can be called
after a successful repository push is made. This includes when a new tag is added
to an existing image.

The webhook call will generate a HTTP POST with the following JSON
payload:

```
{
  "callback_url": "https://registry.hub.docker.com/u/svendowideit/testhook/hook/2141b5bi5i5b02bec211i4eeih0242eg11000a/",
  "push_data": {
    "images": [
        "27d47432a69bca5f2700e4dff7de0388ed65f9d3fb1ec645e2bc24c223dc1cc3",
        "51a9c7c1f8bb2fa19bcd09789a34e63f35abb80044bc10196e304f6634cc582c",
        ...
    ],
    "pushed_at": 1.417566161e+09,
    "pusher": "trustedbuilder"
  },
  "repository": {
    "comment_count": 0,
    "date_created": 1.417494799e+09,
    "description": "",
    "dockerfile": "#\n# BUILD\u0009\u0009docker build -t svendowideit/apt-cacher .\n# RUN\u0009\u0009docker run -d -p 3142:3142 -name apt-cacher-run apt-cacher\n#\n# and then you can run containers with:\n# \u0009\u0009docker run -t -i -rm -e http_proxy http://192.168.1.2:3142/ debian bash\n#\nFROM\u0009\u0009ubuntu\nMAINTAINER\u0009SvenDowideit@home.org.au\n\n\nVOLUME\u0009\u0009[\"/var/cache/apt-cacher-ng\"]\nRUN\u0009\u0009apt-get update ; apt-get install -yq apt-cacher-ng\n\nEXPOSE \u0009\u00093142\nCMD\u0009\u0009chmod 777 /var/cache/apt-cacher-ng ; /etc/init.d/apt-cacher-ng start ; tail -f /var/log/apt-cacher-ng/*\n",
    "full_description": "Docker Hub based automated build from a GitHub repo",
    "is_official": false,
    "is_private": true,
    "is_trusted": true,
    "name": "testhook",
    "namespace": "svendowideit",
    "owner": "svendowideit",
    "repo_name": "svendowideit/testhook",
    "repo_url": "https://registry.hub.docker.com/u/svendowideit/testhook/",
    "star_count": 0,
    "status": "Active"
  }
}
```

Webhooks are available under the Settings menu of each Repository.  
Use a tool like [requestb.in](http://requestb.in/) to test your webhook.

> **Note**: The Docker Hub servers use an elastic IP range, so you can't
> filter requests by IP.

### Webhook chains

Webhook chains allow you to chain calls to multiple services. For example,
you can use this to trigger a deployment of your container only after
it has been successfully tested, then update a separate Changelog once the
deployment is complete.
After clicking the "Add webhook" button, simply add as many URLs as necessary
in your chain.

The first webhook in a chain will be called after a successful push. Subsequent
URLs will be contacted after the callback has been validated.

### Validating a callback

In order to validate a callback in a webhook chain, you need to

1. Retrieve the `callback_url` value in the request's JSON payload.
1. Send a POST request to this URL containing a valid JSON body.

> **Note**: A chain request will only be considered complete once the last
> callback has been validated.

To help you debug or simply view the results of your webhook(s),
view the "History" of the webhook available on its settings page.

### Callback JSON data

The following parameters are recognized in callback data:

* `state` (required): Accepted values are `success`, `failure` and `error`.
  If the state isn't `success`, the webhook chain will be interrupted.
* `description`: A string containing miscellaneous information that will be
  available on the Docker Hub. Maximum 255 characters.
* `context`: A string containing the context of the operation. Can be retrieved
  from the Docker Hub. Maximum 100 characters.
* `target_url`: The URL where the results of the operation can be found. Can be
  retrieved on the Docker Hub.

*Example callback payload:*

    {
      "state": "success",
      "description": "387 tests PASSED",
      "context": "Continuous integration by Acme CI",
      "target_url": "http://ci.acme.com/results/afd339c1c3d27"
    }

## Repository links

Repository links are a way to associate one Automated Build with
another. If one gets updated, the linking system triggers a rebuild
for the other Automated Build. This makes it easy to keep all your
Automated Builds up to date.

To add a link, go to the repository for the Automated Build you want to
link to and click on *Repository Links* under the Settings menu at
right. Then, enter the name of the repository that you want have linked.

> **Warning:**
> You can add more than one repository link, however, you should
> do so very carefully. Creating a two way relationship between Automated Builds will
> cause an endless build loop.
