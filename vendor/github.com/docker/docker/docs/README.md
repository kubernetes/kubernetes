<!--[metadata]>
+++
draft = true
+++
<![end-metadata]-->

# Docker Documentation

The source for Docker documentation is in this directory. Our
documentation uses extended Markdown, as implemented by
[MkDocs](http://mkdocs.org).  The current release of the Docker documentation
resides on [https://docs.docker.com](https://docs.docker.com).

## Understanding the documentation branches and processes

Docker has two primary branches for documentation:

| Branch   | Description                    | URL (published via commit-hook)                                              |
|----------|--------------------------------|------------------------------------------------------------------------------|
| `docs`   | Official release documentation | [https://docs.docker.com](https://docs.docker.com)                             |
| `master` | Merged but unreleased development work    | [http://docs.master.dockerproject.org](http://docs.master.dockerproject.org) |

Additions and updates to upcoming releases are made in a feature branch off of
the `master` branch. The Docker maintainers also support a `docs` branch that
contains the last release of documentation.

After a release, documentation updates are continually merged into `master` as
they occur. This work includes new documentation for forthcoming features, bug
fixes, and other updates. Docker's CI system automatically builds and updates
the `master` documentation after each merge and posts it to
[http://docs.master.dockerproject.org](http://docs.master.dockerproject.org). 

Periodically, the Docker maintainers update `docs.docker.com` between official
releases of Docker. They do this by cherry-picking commits from `master`,
merging them into `docs`,  and then publishing the result.

In the rare case where a change is not forward-compatible, changes may be made
on other branches by special arrangement with the Docker maintainers.

### Quickstart for documentation contributors

If you are a new or beginner contributor, we encourage you to read through the
[our detailed contributors
guide](https://docs.docker.com/project/who-written-for/). The guide explains in
detail, with examples, how to contribute. If you are an experienced contributor
this quickstart should be enough to get you started.

The following is the essential workflow for contributing to the documentation:

1. Fork the `docker/docker` repository.

2. Clone the repository to your local machine.

3. Select an issue from `docker/docker` to work on or submit a proposal of your
own.

4. Create a feature branch from `master` in which to work.

	By basing from `master` your work is automatically included in the next
	release. It also allows docs maintainers to easily cherry-pick your changes
	into the `docs` release branch. 

4. Modify existing or add new `.md` files to the `docs` directory.

	If you add a new document (`.md`) file, you must also add it to the
	appropriate section of the `docs/mkdocs.yml` file in this repository.


5.  As you work, build the documentation site locally to see your changes.

	The `docker/docker` repository contains a `Dockerfile` and a `Makefile`.
	Together, these create a development environment in which you can build and
	run a container running the Docker documentation website. To build the
	documentation site, enter `make docs` at the root of your `docker/docker`
	fork:
	
		$ make docs
		.... (lots of output) ....
		docker run --rm -it  -e AWS_S3_BUCKET -p 8000:8000 "docker-docs:master" mkdocs serve
		Running at: http://0.0.0.0:8000/
		Live reload enabled.
		Hold ctrl+c to quit.
	
	
	The build creates an image containing all the required tools, adds the local
	`docs/` directory and generates the HTML files. Then, it runs a Docker
	container with this image.

	The container exposes port 8000 on the localhost so that you can connect and
	see your changes. If you are running Boot2Docker, use the `boot2docker ip`
	to get the address of your server.

6.  Check your writing for style and mechanical errors.

	Use our [documentation style
	guide](https://docs.docker.com/project/doc-style/) to check style. There are
	several [good grammar and spelling online
	checkers](http://www.hemingwayapp.com/) that can check your writing
	mechanics.

7.  Squash your commits on your branch.

8.  Make a pull request from your fork back to Docker's `master` branch.

9.  Work with the reviewers until your change is approved and merged.

### Debugging and testing

If you have any issues you need to debug, you can use `make docs-shell` and then
run `mkdocs serve`. You can use `make docs-test` to generate a report of missing
links that are referenced in the documentation&mdash;there should be none.

## Style guide

If you have questions about how to write for Docker's documentation, please see
the [style guide](project/doc-style.md). The style guide provides
guidance about grammar, syntax, formatting, styling, language, or tone. If
something isn't clear in the guide, please submit an issue to let us know or
submit a pull request to help us improve it.


## Publishing documentation (for Docker maintainers)

To publish Docker's documentation you need to have Docker up and running on your
machine. You'll also need a `docs/awsconfig` file containing the settings you
need to access the AWS bucket you'll be deploying to.

The process for publishing is to build first to an AWS bucket, verify the build,
and then publish the final release.

1. Have Docker installed and running on your machine.

2. Ask the core maintainers for the `awsconfig` file.

3. Copy the `awsconfig` file to the `docs/` directory.

	The `awsconfig` file contains the profiles of the S3 buckets for our
	documentation sites. (If needed, the release script creates an S3 bucket and
	pushes the files to it.)  Each profile has this format:

		[profile dowideit-docs]
		aws_access_key_id = IHOIUAHSIDH234rwf....
		aws_secret_access_key = OIUYSADJHLKUHQWIUHE......
		region = ap-southeast-2

	The `profile` name must be the same as the name of the bucket you are
	deploying to.

4. Call the `make` from the `docker` directory.

    	$ make AWS_S3_BUCKET=dowideit-docs docs-release

	This publishes _only_ to the `http://bucket-url/v1.2/` version of the
	documentation.

5.  If you're publishing the current release's documentation, you need to also
update the root docs pages by running

     	$ make AWS_S3_BUCKET=dowideit-docs BUILD_ROOT=yes docs-release

### Errors publishing using Boot2Docker

Sometimes, in a Boot2Docker environment, the publishing procedure returns this
error:

	Post http:///var/run/docker.sock/build?rm=1&t=docker-docs%3Apost-1.2.0-docs_update-2:
	dial unix /var/run/docker.sock: no such file or directory.

If this happens, set the Docker host. Run the following command to set the
variables in your shell:

		$ eval "$(boot2docker shellinit)"

## Cherry-picking documentation changes to update an existing release.

Whenever the core team makes a release, they publish the documentation based on
the `release` branch. At that time, the  `release` branch is copied into the
`docs` branch. The documentation team makes updates between Docker releases by
cherry-picking changes from `master` into any of the documentation branches.
Typically, we cherry-pick into the `docs` branch.

For example, to update the current release's docs, do the following:

1. Go to your `docker/docker` fork and get the latest from master.

    	$ git fetch upstream
        
2. Checkout a new branch based on `upstream/docs`.

	You should give your new branch a descriptive name.

		$ git checkout -b post-1.2.0-docs-update-1 upstream/docs
	
3. In a browser window, open [https://github.com/docker/docker/commits/master].

4. Locate the merges you want to publish.

	You should only cherry-pick individual commits; do not cherry-pick merge
	commits. To minimize merge conflicts, start with the oldest commit and work
	your way forward in time.

5. Copy the commit SHA from GitHub.

6. Cherry-pick the commit.
	
	 	$ git cherry-pick -x fe845c4
	
7. Repeat until you have cherry-picked everything you want to merge.

8. Push your changes to your fork.

    	$ git push origin post-1.2.0-docs-update-1

9. Make a pull request to merge into the `docs` branch.

	Do __NOT__ merge into `master`.

10. Have maintainers review your pull request.

11. Once the PR has the needed "LGTMs", merge it on GitHub.

12. Return to your local fork and make sure you are still on the `docs` branch.

		$ git checkout docs

13. Fetch your merged pull request from `docs`.

		$ git fetch upstream/docs
	
14. Ensure your branch is clean and set to the latest.

   	 	$ git reset --hard upstream/docs
    
15. Copy the `awsconfig` file into the `docs` directory.
    
16. Make the beta documentation

    	$ make AWS_S3_BUCKET=beta-docs.docker.io BUILD_ROOT=yes docs-release

17. Open [the beta
website](http://beta-docs.docker.io.s3-website-us-west-2.amazonaws.com/) site
and make sure what you published is correct.

19. When you're happy with your content, publish the docs to our live site:

   		$ make AWS_S3_BUCKET=docs.docker.com BUILD_ROOT=yes
DISTRIBUTION_ID=C2K6......FL2F docs-release

20. Test the uncached version of the live docs at [http://docs.docker.com.s3-website-us-east-1.amazonaws.com/]


### Caching and the docs

New docs do not appear live on the site until the cache (a complex, distributed
CDN system) is flushed. The `make docs-release` command flushes the cache _if_
the `DISTRIBUTION_ID` is set to the Cloudfront distribution ID. The cache flush
can take at least 15 minutes to run and you can check its progress with the CDN
Cloudfront Purge Tool Chrome app.

## Removing files from the docs.docker.com site

Sometimes it becomes necessary to remove files from the historical published documentation.
The most reliable way to do this is to do it directly using `aws s3` commands running in a
docs container:

Start the docs container like `make docs-shell`, but bind mount in your `awsconfig`:

```
docker run --rm -it -v $(CURDIR)/docs/awsconfig:/docs/awsconfig docker-docs:master bash
```

and then the following example shows deleting 2 documents from s3, and then requesting the
CloudFlare cache to invalidate them:


```
export BUCKET BUCKET=docs.docker.com
export AWS_CONFIG_FILE=$(pwd)/awsconfig
aws s3 --profile $BUCKET ls s3://$BUCKET
aws s3 --profile $BUCKET rm s3://$BUCKET/v1.0/reference/api/docker_io_oauth_api/index.html
aws s3 --profile $BUCKET rm s3://$BUCKET/v1.1/reference/api/docker_io_oauth_api/index.html

aws configure set preview.cloudfront true
export DISTRIBUTION_ID=YUTIYUTIUTIUYTIUT
aws cloudfront  create-invalidation --profile docs.docker.com --distribution-id $DISTRIBUTION_ID --invalidation-batch '{"Paths":{"Quantity":1, "Items":["/v1.0/reference/api/docker_io_oauth_api/"]},"CallerReference":"6Mar2015sventest1"}'
aws cloudfront  create-invalidation --profile docs.docker.com --distribution-id $DISTRIBUTION_ID --invalidation-batch '{"Paths":{"Quantity":1, "Items":["/v1.1/reference/api/docker_io_oauth_api/"]},"CallerReference":"6Mar2015sventest1"}'
```

### Generate the man pages 

For information on generating man pages (short for manual page), see [the man
page directory](https://github.com/docker/docker/tree/master/docker) in this
project.




