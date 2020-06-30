## Changing API

OpenShift is split into three major repositories:

1. https://github.com/openshift/api/ - which holds all the external API objects definitions.
1. https://github.com/openshift/client-go/ - which holds all the client code (written in Go).
1. https://github.com/openshift/origin/ - which holds the actual code behind OpenShift.

This split requires additional effort to introduce any API change.  The following steps
should guide you through the process.

1. The first place to introduce the changes is [openshift/api](https://github.com/openshift/api/).
Here, you put your external API updates and when you are done run `make generate`.  If you
need to introduce a new dependency run `make update-deps`, and almost never update `glide.yaml`
directly.  When you're done open a PR against the aforementioned repository and ping
[@openshift/api-review](https://github.com/orgs/openshift/teams/api-review) for a review.

2. The next step includes updating the [openshift/client-go](https://github.com/openshift/client-go/)
with the changes from step 1, since it vendors it.  To do so run `make update-deps` to pick up
the changes from step 1 and then run `make generate` to update the client code with necessary
changes.  When you're done open a PR against the aforementioned repository and ping
[@openshift/sig-master](https://github.com/orgs/openshift/teams/sig-master) for a review.

3. The final step happens in [openshift/origin](https://github.com/openshift/origin/) repository.
As previously, run `make update-deps` to pick up the changes from previous two steps.  Afterwards,
run `make update` to generate the remaining bits in origin repository. When you're done open
a PR against the aforementioned repository and ping [@openshift/sig-master](https://github.com/orgs/openshift/teams/sig-master)
for a review.

If at any point you have doubts about any step of the flow reach out to
[@openshift/sig-master](https://github.com/orgs/openshift/teams/sig-master) team for help.

NOTE: It may happen that during `make update-deps` step you will pick up the changes introduced
by someone else in his PR.  In that case sync with the other PR's author and include his changes
in your PR noting the fact to your reviewer.

## Cherry-picking an upstream commit into Origin: Why, how, and when.

Origin carries patches inside of vendor/ on top of each rebase.
Thus, origin carries upstream patches in two ways.

1. *periodic rebases* against a Kubernetes commit.
Eventually, any code you have in upstream kubernetes will land in Openshift
via this mechanism.

2. Cherry-picked patches for important *bug fixes*.  We really try to
limit feature back-porting entirely.

### Manually

You can manually try to cherry pick a commit (by using git apply). This can
easily be done in a couple of steps.

- wget the patch, i.e. `wget -O /tmp/mypatch
https://github.com/kubernetes/kubernetes/pull/34624.patch`
- PATCH=/tmp/mypatch git apply --directory vendor/k8s.io/kubernetes $PATCH

If this fails, then it's possible you may need to pick multiple commits.

### For Openshift newcomers: Pick my kubernetes fix into Openshift vs. wait for the next rebase?

Assuming you read the bullets above... If your patch is really far behind, for
example, if there have been 5 commits modifying the directory you care about,
cherry picking will be increasingly difficult and you should consider waiting
for the next rebase, which will likely include the commit you care about or at
least decrease the amount of cherry picks you need to do to merge.

To really know the answer, you need to know *how many commits behind you are in
a particular directory*, often.

To do this, just use git log, like so (using pkg/scheduler/ as an example).

```
MYDIR=pkg/scheduler/algorithm git log --oneline --
vendor/k8s.io/kubernetes/${MYDIR} | grep UPSTREAM | cut -d' ' -f 4-10 | head -1
```

The commit message printed above will tell you:

- what the LAST commit in Kubernetes was (which effected
"/pkg/scheduler/algorithm")
- directory, which will give you an intuition about how "hot" the code you are
cherry picking is.  If it has changed a lot, recently, then that means you
probably will want to wait for a rebase to land.

### Using hack/cherry-pick

For convenience, you can use `hack/cherry-pick.sh` to generate patches for
Origin from upstream commits.

The purpose of this command is to allow you to pull individual commits from a
local kubernetes repository into origin's vendored kuberenetes in a fully
automated manner.

To use this command, be sure to setup remote Pull Request branches in the
kubernetes repository you are using (i.e. like https://gist.github.com/piscisaureus/3342247).
Specifically, you will be doing this, to the git config you probably already
have for kubernetes:

```
[remote "origin"]
        url = https://github.com/kubernetes/kubernetes
        fetch = +refs/heads/*:refs/remotes/origin/*
	### Add this line
        fetch = +refs/pull/*/head:refs/remotes/origin/pr/*
```

so that `git show origin/pr/<number>` displays information about your branch
after a `git fetch`.

You must also have the Kubernetes repository checked out in your GOPATH
(visible as `../../../k8s.io/kubernetes`),
with openshift/kubernetes as a remote and fetched:

    $ pushd $GOPATH/src/k8s.io/kubernetes
    $ git remote add openshift https://github.com/openshift/kubernetes.git
    $ git fetch openshift
    $ popd

There must be no modified or uncommitted files in either repository.

To pull an upstream commit, run:

    $ hack/cherry-pick.sh <pr_number>

This will attempt to create a patch from the current Kube rebase version in
Origin that contains the commits added in the PR. If the PR has already been
merged into the Kube version, you'll get an error. If there are conflicts, you'll
have to resolve them in the upstream repo, then hit ENTER to continue. The end
result will be a single commit in your Origin repo that contains the changes.

If you want to run without a rebase option, set `NO_REBASE=1` before the
command is run. You can also specify a commit range directly with:

    $ hack/cherry-pick.sh origin/master...<some_branch>

All upstream commits should have a commit message where the first line is:

    UPSTREAM: <PR number|drop|carry|00000>: <short description>

`drop` indicates the commit should be removed during the next
rebase. `carry` means that the change cannot go into upstream, and we
should continue to use it during the next rebase. `PR number` means
that the commit will be dropped during a rebase, as long as that
rebase includes the given PR number. `00000` means that the master team
has opted into carrying the debt until the next rebase when we will attempt
to gather them and create upstream patches.

By default `hack/cherry-pick.sh` uses git remote named `origin` to fetch
kubernetes repository, if your git configuration is different, you can pass the git
remote name by setting `UPSTREAM_REMOTE` env var:

    $ UPSTREAM_REMOTE=upstream hack/cherry-pick.sh <pr_number>

## Moving a commit you developed in Origin to an upstream

The `hack/move-upstream.sh` script takes the current feature branch, finds any
changes to the
requested upstream project (as defined by `UPSTREAM_REPO` and
`UPSTREAM_PACKAGE`) that differ from `origin/master`, and then creates a new
commit in that upstream project on a branch with the same name as your current
branch.

For example, to upstream a commit to OpenShift source-to-image while working
from Origin:

    $ git checkout my_feature_branch_in_origin
    $ git log --oneline
    70ffe7e Docker and STI builder support binary extraction
    75a22de UPSTREAM: <sti>: Allow prepared directories to be passed to STI
    86eefdd UPSTREAM: 14618: Refactor exec to allow reuse from server

    # we want to move our STI changes to upstream
    $ UPSTREAM_REPO=github.com/openshift/source-to-image
UPSTREAM_PACKAGE=pkg/api hack/move-upstream.sh
    ...

    # All changes to source-to-image in Godeps/. are now in a commit UPSTREAMED
in s2i repo

    $ cd ../source-to-image
    $ git log --oneline
    c0029f6 UPSTREAMED
    ... # older commits

The default is to work against Kube.
go

## Updating Kubernetes from upstream

There are a few steps involved in rebasing Origin to a new version of
Kubernetes. We need to make sure
that not only the Kubernetes packages were updated correctly into `Godeps`, but
also that *all tests are
still running without errors* and *code changes, refactorings or the
inclusion/removal of attributes
were properly reflected* in the Origin codebase.

### 1. Preparation

Before you begin, make sure you have both
[openshift/origin](https://github.com/openshift/origin) and
[kubernetes/kubernetes](https://github.com/kubernetes/kubernetes) in your
$GOPATH. You may want to work
on a separate $GOPATH just for the rebase:

```
$ go get github.com/openshift/origin
$ go get k8s.io/kubernetes
```

You must add the Origin GitHub fork as a remote in your k8s.io/kubernetes repo:

```
$ cd $GOPATH/src/k8s.io/kubernetes
$ git remote add openshift git@github.com:openshift/kubernetes.git
$ git fetch openshift
```

Check out the version of Kubernetes you want to rebase as a branch or tag named
`stable_proposed` in
[kubernetes/kubernetes](https://github.com/kubernetes/kubernetes). For example,
if you are going to rebase the latest `master` of Kubernetes:

```
$ cd $GOPATH/src/k8s.io/kubernetes
$ git checkout master
$ git pull
$ git checkout -b stable_proposed
```

### 2. Rebase Origin to the new Kubernetes version

#### 2.1. First option (preferred): using the rebase-kube.sh script

If all requirements described in *Preparation* were correctly attended, you
should not have any trouble with rebasing the Kubernetes code using the script
that automates this process.

```
$ cd $GOPATH/src/github.com/openshift/origin
$ hack/rebase-kube.sh
```

Read over the changes with `git status` and make sure it looks reasonable.
Check specially the `Godeps/Godeps.json` file to make sure no dependency
is unintentionally missing.

Commit using the message `bump(k8s.io/kubernetes):<commit SHA>`, where
`<commit SHA>` is the commit id for the Kubernetes version we are including in
our Godeps. It can be found in our `Godeps/Godeps.json` in the declaration of
any Kubernetes package.

#### 2.2. Second option: manually

If for any reason you had trouble rebasing using the script, you may need to to
do it manually.
After following all requirements described in the *Preparation* topic, you will
need to run
`godep restore` from both the Origin and the Kubernetes directories and then
`godep save ./...`
from the Origin directory. Follow these steps:

1. `$ cd $GOPATH/src/github.com/openshift/origin`
2. `make clean ; godep restore` will restore the package versions specified in
the `Godeps/Godeps.json` of Origin to your GOPATH.
2. `$ cd $GOPATH/src/k8s.io/kubernetes`
3. `$ git checkout stable_proposed` will checkout the desired version of
Kubernetes as branched in *Preparation*.
4. `$ godep restore` will restore the package versions specified in the
`Godeps/Godeps.json` of Kubernetes to your GOPATH.
5. `$ cd $GOPATH/src/github.com/openshift/origin`.
6. `$ make clean ; godep save ./...` will save a list of the checked-out
dependencies to the file `Godeps/Godeps.json`, and copy their source code into `vendor`.
7. If in the previous step godep complaints about the checked out revision of a
package being different  than the wanted revision, this probably means there are
new packages in Kubernetes that we need to add.  Do a `godep save <pkgname>` with
the package specified by the error message and then `$ godep save ./...`
again.
8. Read over the changes with `git status` and make sure it looks reasonable.
Check specially the `Godeps/Godeps.json` file to make sure no dependency is
unintentionally missing. The whole Godeps directory will be added to version control, including
`_workspace`.
9. Commit using the message `bump(k8s.io/kubernetes):<commit SHA>`, where
`<commit SHA>` is the commit id for the Kubernetes version we are including in
our Godeps. It can be found in our `Godeps/Godeps.json` in the declaration of
any Kubernetes package.

If in the process of rebasing manually you found any corner case not attended
by the `hack/rebase-kube.sh` script, make sure you update it accordingly to help future rebases.

### 3. cherry-pick upstream changes pushed to the Origin repo

Eventually, during the development cycle, we introduce changes to dependencies
right in the Origin
repository. This is not a largely recommended practice, but it's useful if we
need something that,
for example, is in the Kubernetes repository but we are not doing a rebase yet.
So, when doing the next
rebase, we need to make sure we get all these changes otherwise they will be
overridden by `godep save`.

1. Check the `Godeps` directory [commits
history](https://github.com/openshift/origin/commits/master/Godeps)
for commits tagged with the *UPSTREAM* keyword. We will need to cherry-pick
*all UPSTREAM commits since
the last Kubernetes rebase* (remember you can find the last rebase commit
looking for a message like
`bump(k8s.io/kubernetes):...`).
2. For every commit tagged UPSTREAM, do `git cherry-pick <commit SHA>`.
3. Notice that eventually the cherry-pick will be empty. This probably means
the given change was
already merged in Kubernetes and we don't need to specifically add it to our
Godeps. Nice!
4. Read over the commit history and make sure you have every UPSTREAM commit
since the last rebase
(except only for the empty ones).

### 4. Refactor Origin to be compliant with upstream changes

After making sure we have all the dependencies in place and up-to-date, we need
to work in the Origin
codebase to make sure the compilation is not broken, all tests pass and it's
compliant with any refactorings, architectural changes or behavior changes
introduced in Kubernetes. Make sure:

1. `make clean ; hack/build-go.sh` compiles without errors and the standalone
server starts correctly.
2. all of our generated code is up to date by running all `hack/update-*`
scripts.
3. `hack/verify-open-ports.sh` runs without errors.
4. `hack/copy-kube-artifacts.sh` so Kubernetes tests can be fully functional.
The diff resulting from this script should be squashed into the Kube bump
commit.
5. `TEST_KUBE=1 hack/test-go.sh` runs without errors.
6. `hack/test-cmd.sh` runs without errors.
7. `hack/test-integration.sh` runs without errors.
8. `hack/test-end-to-end.sh` runs without errors.
    See *Building a Release* above for setting up the environment for the
*test-end-to-end.sh* tests.

It is helpful to look at the Kubernetes commit history to be aware of the major
topics. Although it
can potentially break or change any part of Origin, the most affected parts are
usually:

1. https://github.com/openshift/origin/blob/master/pkg/cmd/server/start
2. https://github.com/openshift/origin/blob/master/pkg/cmd/server/origin/master.go
3. https://github.com/openshift/origin/blob/master/pkg/oc/cli/util/clientcmd/factory.go
4. https://github.com/openshift/origin/blob/master/pkg/oc/cli/cli.go
5. https://github.com/openshift/origin/blob/master/pkg/api/meta/meta.go

Place all your changes in a commit called "Refactor to match changes upstream".

### 5. Pull request

A typical pull request for your Kubernetes rebase will contain:

1. One commit for the Kuberentes Godeps bump (`bump(k8s.io/kubernetes):<commit
SHA>`).
2. Zero, one, or more bump commits for any **shared** dependencies between
Origin and Kubernetes that have been bumped. Any transitive dependencies coming
from Kubernetes should be squashed in the Kube bump commit.
3. Zero, one, or more cherry-picked commits tagged UPSTREAM.
4. One commit "Boring refactor to match changes upstream" that includes boring
changes like imports rewriting, etc.
5. One commit "Interesting refactor to match changes upstream" that includes
interesting changes like new plugins or controller changes.

## RPM Packaging

A specfile is included in this repo which can be used to produce RPMs including
the openshift binary. While the specfile will be kept up to date with build
requirements the version is not updated. You will need to either update the
Version, %commit, and %ldflags values on your own or you may use
[tito](https://github.com/dgoodwin/tito) to build
and tag releases.

## Troubleshooting

If you run into difficulties running OpenShift, start by reading through the
[troubleshooting
guide](https://github.com/openshift/origin/blob/master/docs/debugging-openshift.
md).

## Swagger API Documentation

OpenShift and Kubernetes integrate with the [Swagger 2.0 API
framework](http://swagger.io) which aims to make it easier to document and
write clients for RESTful APIs.  When you start OpenShift, the Swagger API
endpoint is exposed at `https://localhost:8443/swaggerapi`. The Swagger UI
makes it easy to view your documentation - to view the docs for your local
version of OpenShift start the server with CORS enabled and then browse to
http://openshift3swagger-claytondev.rhcloud.com (which runs a copy of the
Swagger UI that points to localhost:8080 by default).  Expand the operations
available on v1 to see the schemas (and to try the API directly).
Additionally, you can download swagger-ui from http://swagger.io/swagger-ui/
and use it to point to your local swagger API endpoint.

Note: Hosted API documentation can be found
[here](http://docs.okd.io/latest/rest_api/openshift_v1.html).

## Performance debugging

OpenShift integrates the go `pprof` tooling to make it easy to capture CPU and
heap dumps for running systems.  The pprof endpoint is available at `/debug/pprof/`
on the secured HTTPS port for the `openshift` binary:

    $ oc get --raw /debug/pprof/profile --as=system:admin > cpu.pprof

To view profiles, you use
[pprof](http://goog-perftools.sourceforge.net/doc/cpu_profiler.html) which is
part of `go tool`.  You must pass the captured pprof file (for source lines
you will need to build the binary locally).  For instance, to view a `cpu` profile
from above, you would run OpenShift to completion, and then run:

    $ go tool pprof cpu.pprof
    or
    $ go tool pprof /var/lib/origin/cpu.pprof

This will open the `pprof` shell, and you can then run:

    # see the top 20 results
    (pprof) top20

    # see the top 50 results
    (pprof) top50

    # show the top20 sorted by cumulative time
    (pprof) cum=true
    (pprof) top20

to see the top20 CPU consuming fields or

    (pprof) web

to launch a web browser window showing you where CPU time is going.

`pprof` supports CLI arguments for looking at profiles in different ways -
memory profiles by default show allocated space:

    $ go tool pprof mem.pprof

but you can also see the allocated object counts:

    $ go tool pprof --alloc_objects mem.pprof

Finally, when using the `web` profile mode, you can have the go tool directly
fetch your profiles via HTTP for services that only expose their profiling
contents over an unsecured HTTP endpoint:

    # for a 30s CPU trace
    $ go tool pprof http://127.0.0.1:6060/debug/pprof/profile

    # for a snapshot heap dump at the current time, showing total allocations
    $ go tool pprof --alloc_space ./_output/local/bin/linux/amd64/openshift
http://127.0.0.1:6060/debug/pprof/heap

See [debugging Go programs](https://golang.org/pkg/net/http/pprof/) for more
info.  `pprof` has many modes and is very powerful (try `tree`) - you can pass
a regex to many arguments to limit your results to only those samples that
match the regex (basically the function name or the call stack).
