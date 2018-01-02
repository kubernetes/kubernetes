# Tools

This page describes the tools we use and infrastructure that is in place for
the Docker project.

### CI

The Docker project uses [Jenkins](https://jenkins.dockerproject.org/) as our
continuous integration server. Each Pull Request to Docker is tested by running the 
equivalent of `make all`. We chose Jenkins because we can host it ourselves and
we run Docker in Docker to test.

#### Leeroy

Leeroy is a Go application which integrates Jenkins with 
GitHub pull requests. Leeroy uses 
[GitHub hooks](https://developer.github.com/v3/repos/hooks/) 
to listen for pull request notifications and starts jobs on your Jenkins 
server.  Using the Jenkins
[notification plugin](https://wiki.jenkins-ci.org/display/JENKINS/Notification+Plugin),
Leeroy updates the pull request using GitHub's 
[status API](https://developer.github.com/v3/repos/statuses/)
with pending, success, failure, or error statuses.

The leeroy repository is maintained at
[github.com/docker/leeroy](https://github.com/docker/leeroy).

#### GordonTheTurtle IRC Bot

The GordonTheTurtle IRC Bot lives in the
[#docker-maintainers](https://botbot.me/freenode/docker-maintainers/) channel
on Freenode. He is built in Go and is based off the project at
[github.com/fabioxgn/go-bot](https://github.com/fabioxgn/go-bot). 

His main command is `!rebuild`, which rebuilds a given Pull Request for a repository.
This command works by integrating with Leroy. He has a few other commands too, such 
as `!gif` or `!godoc`, but we are always looking for more fun commands to add.

The gordon-bot repository is maintained at
[github.com/docker/gordon-bot](https://github.com/docker/gordon-bot)

### NSQ

We use [NSQ](https://github.com/bitly/nsq) for various aspects of the project
infrastructure.

#### Hooks

The hooks project,
[github.com/crosbymichael/hooks](https://github.com/crosbymichael/hooks),
is a small Go application that manages web hooks from github, hub.docker.com, or
other third party services.

It can be used for listening to github webhooks & pushing them to a queue,
archiving hooks to rethinkdb for processing, and broadcasting hooks to various
jobs.

#### Docker Master Binaries

One of the things queued from the Hooks are the building of the Master
Binaries. This happens on every push to the master branch of Docker. The
repository for this is maintained at
[github.com/docker/docker-bb](https://github.com/docker/docker-bb).
