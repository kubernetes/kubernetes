<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Design: Indexed Feature of Job object


## Summary

This design extends kubernetes with user-friendly support for
running embarrassingly parallel jobs.

Here, *parallel* means on multiple nodes, which means multiple pods.
By *embarrassingly parallel*,  it is meant that the pods
have no dependencies between each other.  In particular, neither
ordering between pods nor gang scheduling are supported.

Users already have two other options for running embarrassingly parallel
Jobs (described in the next section), but both have ease-of-use issues.

Therefore, this document proposes extending the Job resource type to support
a third way to run embarrassingly parallel programs, with a focus on
ease of use.

This new style of Job is called an *indexed job*, because each Pod of the Job
is specialized to work on a particular *index* from a fixed length array of work
items.

## Background

The Kubernetes [Job](../../docs/user-guide/jobs.md) already supports
the embarrassingly parallel use case through *workqueue jobs*.
While [workqueue jobs](../../docs/user-guide/jobs.md#job-patterns) are very
flexible, they can be difficult to use. They: (1) typically require running a
message queue or other database service, (2) typically require modifications
to existing binaries and images and (3) subtle race conditions are easy to
 overlook.

Users also have another option for parallel jobs: creating [multiple Job objects
from a template](hdocs/design/indexed-job.md#job-patterns). For small numbers of
Jobs, this is a fine choice. Labels make it easy to view and delete multiple Job
objects at once. But, that approach also has its drawbacks: (1) for large levels
of parallelism (hundreds or thousands of pods) this approach means that listing
all jobs presents too much information, (2) users want a single source of
information about the success or failure of what the user views as a single
logical process.

Indexed job fills provides a third option with better ease-of-use for common
use cases.

## Requirements

### User Requirements

- Users want an easy way to run a Pod to completion *for each* item within a
[work list](#example-use-cases).

- Users want to run these pods in parallel for speed, but to vary the level of
parallelism as needed, independent of the number of work items.

- Users want to do this without requiring changes to existing images,
or source-to-image pipelines.

- Users want a single object that encompasses the lifetime of the parallel
program. Deleting it should delete all dependent objects. It should report the
status of the overall process. Users should be able to wait for it to complete,
and can refer to it from other resource types, such as
[ScheduledJob](https://github.com/kubernetes/kubernetes/pull/11980).


### Example Use Cases

Here are several examples of *work lists*: lists of command lines that the user
wants to run, each line its own Pod. (Note that in practice, a work list may not
ever be written out in this form, but it exists in the mind of the Job creator,
and it is a useful way to talk about the intent of the user when discussing
alternatives for specifying Indexed Jobs).

Note that we will not have the user express their requirements in work list
form; it is just a format for presenting use cases. Subsequent discussion will
reference these work lists.

#### Work List 1

Process several files with the same program:

```
/usr/local/bin/process_file 12342.dat
/usr/local/bin/process_file 97283.dat
/usr/local/bin/process_file 38732.dat
```

#### Work List 2

Process a matrix (or image, etc) in rectangular blocks:

```
/usr/local/bin/process_matrix_block -start_row 0 -end_row 15 -start_col 0 --end_col 15
/usr/local/bin/process_matrix_block -start_row 16 -end_row 31 -start_col 0 --end_col 15
/usr/local/bin/process_matrix_block -start_row 0 -end_row 15 -start_col 16 --end_col 31
/usr/local/bin/process_matrix_block -start_row 16 -end_row 31 -start_col 16 --end_col 31
```

#### Work List 3

Build a program at several different git commits:

```
HASH=3cab5cb4a git checkout $HASH && make clean && make VERSION=$HASH
HASH=fe97ef90b git checkout $HASH && make clean && make VERSION=$HASH
HASH=a8b5e34c5 git checkout $HASH && make clean && make VERSION=$HASH
```

#### Work List 4

Render several frames of a movie:

```
./blender /vol1/mymodel.blend -o /vol2/frame_#### -f 1
./blender /vol1/mymodel.blend -o /vol2/frame_#### -f 2
./blender /vol1/mymodel.blend -o /vol2/frame_#### -f 3
```

#### Work List 5

Render several blocks of frames (Render blocks to avoid Pod startup overhead for
every frame):

```
./blender /vol1/mymodel.blend -o /vol2/frame_#### --frame-start 1 --frame-end 100
./blender /vol1/mymodel.blend -o /vol2/frame_#### --frame-start 101 --frame-end 200
./blender /vol1/mymodel.blend -o /vol2/frame_#### --frame-start 201 --frame-end 300
```

## Design Discussion

### Converting Work Lists into Indexed Jobs.

Given a work list, like in the [work list examples](#work-list-examples),
the information from the work list needs to get into each Pod of the Job.

Users will typically not want to create a new image for each job they
run. They will want to use existing images. So, the image is not the place
for the work list.

A work list can be stored on networked storage, and mounted by pods of the job.
Also, as a shortcut, for small worklists, it can be included in an annotation on
the Job object, which is then exposed as a volume in the pod via the downward
API.

### What Varies Between Pods of a Job

Pods need to differ in some way to do something different. (They do not differ
in the work-queue style of Job, but that style has ease-of-use issues).

A general approach would be to allow pods to differ from each other in arbitrary
ways. For example, the Job object could have a list of PodSpecs to run.
However, this is so general that it provides little value. It would:

- make the Job Spec very verbose, especially for jobs with thousands of work
items
- Job becomes such a vague concept that it is hard to explain to users
- in practice, we do not see cases where many pods which differ across many
fields of their specs, and need to run as a group, with no ordering constraints.
- CLIs and UIs need to support more options for creating Job
- it is useful for monitoring and accounting databases want to aggregate data
for pods with the same controller. However, pods with very different Specs may
not make sense to aggregate.
- profiling, debugging, accounting, auditing and monitoring tools cannot assume
common images/files, behaviors, provenance and so on between Pods of a Job.

Also, variety has another cost. Pods which differ in ways that affect scheduling
(node constraints, resource requirements, labels) prevent the scheduler from
treating them as fungible, which is an important optimization for the scheduler.

Therefore, we will not allow Pods from the same Job to differ arbitrarily
(anyway, users can use multiple Job objects for that case).  We will try to
allow as little as possible to differ between pods of the same Job, while still
allowing users to express common parallel patterns easily. For users who need to
run jobs which differ in other ways, they can create multiple Jobs, and manage
them as a group using labels.

From the above work lists, we see a need for Pods which differ in their command
lines, and in their environment variables.  These work lists do not require the
pods to differ in other ways.

Experience in [similar systems](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43438.pdf)
has shown this model to be applicable to a very broad range of problems, despite
this restriction.

Therefore we to allow pods in the same Job to differ **only** in the following
 aspects:
- command line
- environment variables

### Composition of existing images

The docker image that is used in a job may not be maintained by the person
running the job.  Over time, the Dockerfile may change the ENTRYPOINT or CMD.
If we require people to specify the complete command line to use Indexed Job,
then they will not automatically pick up changes in the default
command or args.

This needs more thought.

### Running Ad-Hoc Jobs using kubectl

A user should be able to easily start an Indexed Job using `kubectl`. For
example to run [work list 1](#work-list-1), a user should be able to type
something simple like:

```
kubectl run process-files --image=myfileprocessor \
   --per-completion-env=F="12342.dat 97283.dat 38732.dat" \
   --restart=OnFailure  \
   -- \
   /usr/local/bin/process_file '$F'
```

In the above example:

- `--restart=OnFailure` implies creating a job instead of replicationController.
- Each pods command line is `/usr/local/bin/process_file $F`.
- `--per-completion-env=` implies the jobs `.spec.completions` is set to the
length of the argument array (3 in the example).
- `--per-completion-env=F=<values>` causes env var with `F` to be available in
the environment when the command line is evaluated.

How exactly this happens is discussed later in the doc: this is a sketch of the
user experience.

In practice, the list of files might be much longer and stored in a file on the
users local host, like:

```
$ cat files-to-process.txt
12342.dat
97283.dat
38732.dat
...
```

So, the user could specify instead: `--per-completion-env=F="$(cat files-to-process.txt)"`.

However, `kubectl` should also support a format like:
 `--per-completion-env=F=@files-to-process.txt`.
That allows `kubectl` to parse the file, point out any syntax errors, and would
not run up against command line length limits (2MB is common, as low as 4kB is
POSIX compliant).

One case we do not try to handle is where the file of work is stored on a cloud
filesystem, and not accessible from the users local host.  Then we cannot easily
use indexed job, because we do not know the number of completions.  The user
needs to copy the file locally first or use the Work-Queue style of Job (already
supported).

Another case we do not try to handle is where the input file does not exist yet
because this Job is to be run at a future time, or depends on another job. The
workflow and scheduled job proposal need to consider this case. For that case,
you could use an indexed job which runs a program which shards the input file
(map-reduce-style).

#### Multiple parameters

The user may also have multiple parameters, like in [work list 2](#work-list-2).
One way is to just list all the command lines already expanded, one per line, in
a file, like this:

```
$ cat matrix-commandlines.txt
/usr/local/bin/process_matrix_block -start_row 0 -end_row 15 -start_col 0 --end_col 15
/usr/local/bin/process_matrix_block -start_row 16 -end_row 31 -start_col 0 --end_col 15
/usr/local/bin/process_matrix_block -start_row 0 -end_row 15 -start_col 16 --end_col 31
/usr/local/bin/process_matrix_block -start_row 16 -end_row 31 -start_col 16 --end_col 31
```

and run the Job like this:

```
kubectl run process-matrix --image=my/matrix \
   --per-completion-env=COMMAND_LINE=@matrix-commandlines.txt \
   --restart=OnFailure  \
   -- \
   'eval "$COMMAND_LINE"'
```

However, this may have some subtleties with shell escaping.  Also, it depends on
the user knowing all the correct arguments to the docker image being used (more
on this later).

Instead, kubectl should support multiple instances of the `--per-completion-env`
flag. For example, to implement work list 2, a user could do:

```
kubectl run process-matrix --image=my/matrix \
   --per-completion-env=SR="0 16 0 16" \
   --per-completion-env=ER="15 31 15 31" \
   --per-completion-env=SC="0 0 16 16" \
   --per-completion-env=EC="15 15 31 31" \
   --restart=OnFailure  \
   -- \
   /usr/local/bin/process_matrix_block -start_row $SR -end_row $ER -start_col $ER --end_col $EC 
```

### Composition With Workflows and ScheduledJob

A user should be able to create a job (Indexed or not) which runs at a specific
time(s). For example:

```
$ kubectl run process-files --image=myfileprocessor \
   --per-completion-env=F="12342.dat 97283.dat 38732.dat" \
   --restart=OnFailure  \
   --runAt=2015-07-21T14:00:00Z
   -- \
   /usr/local/bin/process_file '$F'
created "scheduledJob/process-files-37dt3"
```

Kubectl should build the same JobSpec, and then put it into a ScheduledJob
(#11980) and create that.

For [workflow type jobs](../../docs/user-guide/jobs.md#job-patterns), creating a
complete workflow from a single command line would be messy, because of the need
to specify all the arguments multiple times.

For that use case, the user could create a workflow message by hand. Or the user
could create a job template, and then make a workflow from the templates,
perhaps like this:

```
$ kubectl run process-files --image=myfileprocessor \
   --per-completion-env=F="12342.dat 97283.dat 38732.dat" \
   --restart=OnFailure  \
   --asTemplate \
   -- \
   /usr/local/bin/process_file '$F'
created "jobTemplate/process-files"
$ kubectl run merge-files --image=mymerger \
   --restart=OnFailure  \
   --asTemplate \
   -- \
   /usr/local/bin/mergefiles 12342.out 97283.out 38732.out \
created "jobTemplate/merge-files"
$ kubectl create-workflow process-and-merge \
   --job=jobTemplate/process-files
   --job=jobTemplate/merge-files
   --dependency=process-files:merge-files
created "workflow/process-and-merge"
```

### Completion Indexes

A JobSpec specifies the number of times a pod needs to complete successfully,
through the `job.Spec.Completions` field. The number of completions will be
equal to the number of work items in the work list.

Each pod that the job controller creates is intended to complete one work item
from the work list. Since a pod may fail, several pods may, serially, attempt to
complete the same index. Therefore, we call it a *completion index* (or just
*index*), but not a *pod index*.

For each completion index, in the range 1 to `.job.Spec.Completions`, the job
controller will create a pod with that index, and keep creating them on failure,
until each index is completed.

An dense integer index, rather than a sparse string index (e.g. using just
`metadata.generate-name`) makes it easy to use the index to lookup parameters
in, for example, an array in shared storage.

### Pod Identity and Template Substitution in Job Controller

The JobSpec contains a single pod template.  When the job controller creates a
particular pod, it copies the pod template and modifies it in some way to make
that pod distinctive. Whatever is distinctive about that pod is its *identity*.

We consider several options.

#### Index Substitution Only

The job controller substitutes only the *completion index* of the pod into the
pod template when creating it.  The JSON it POSTs differs only in a single
fields.

We would put the completion index as a stringified integer, into an annotation
of the pod. The user can extract it from the annotation into an env var via the
downward API, or put it in a file via a Downward API volume, and parse it
himself.

Once it is an environment variable in the pod (say `$INDEX`), then one of two
things can happen.

First, the main program can know how to map from an integer index to what it
needs to do. For example, from Work List 4 above:

```
./blender /vol1/mymodel.blend -o /vol2/frame_#### -f $INDEX
```

Second, a shell script can be prepended to the original command line which maps
the index to one or more string parameters. For example, to implement Work List
5 above, you could do:

```
/vol0/setupenv.sh && ./blender /vol1/mymodel.blend -o /vol2/frame_#### --frame-start $START_FRAME --frame-end $END_FRAME
```

In the above example, `/vol0/setupenv.sh` is a shell script that reads `$INDEX`
and exports `$START_FRAME` and `$END_FRAME`.

The shell could be part of the image, but more usefully, it could be generated
by a program and stuffed in an annotation or a configMap, and from there added
to a volume.

The first approach may require the user to modify an existing image (see next
section) to be able to accept an `$INDEX` env var or argument. The second
approach requires that the image have a shell. We think that together these two
options cover a wide range of use cases (though not all).

#### Multiple Substitution

In this option, the JobSpec is extended to include a list of values to
substitute, and which fields to substitute them into. For example, a worklist
like this:

```
FRUIT_COLOR=green process-fruit -a -b -c -f apple.txt --remove-seeds
FRUIT_COLOR=yellow process-fruit -a -b -c -f banana.txt
FRUIT_COLOR=red process-fruit -a -b -c -f cherry.txt --remove-pit
```

Can be broken down into a template like this, with three parameters:

```
<custom env var 1>; process-fruit -a -b -c <custom arg 1> <custom arg 1>
```

and a list of parameter tuples, like this:

```
("FRUIT_COLOR=green", "-f apple.txt", "--remove-seeds")
("FRUIT_COLOR=yellow", "-f banana.txt", "")
("FRUIT_COLOR=red", "-f cherry.txt", "--remove-pit")
```

The JobSpec can be extended to hold a list of parameter tuples (which are more
easily expressed as a list of lists of individual parameters). For example:

```
apiVersion: extensions/v1beta1
kind: Job
...
spec:
  completions: 3
  ...
  template:
    ...
  perCompletionArgs:
    container: 0
      -
        - "-f apple.txt"
        - "-f banana.txt"
        - "-f cherry.txt"
      -
        - "--remove-seeds"
        - ""
        - "--remove-pit"
  perCompletionEnvVars:
    - name: "FRUIT_COLOR"
      - "green"
      - "yellow"
      - "red"
```

However, just providing custom env vars, and not arguments, is sufficient for
many use cases: parameter can be put into env vars, and then substituted on the
command line.

#### Comparison

The multiple substitution approach:

- keeps the *per completion parameters* in the JobSpec.
- Drawback: makes the job spec large for job with thousands of completions. (But
for very large jobs, the work-queue style or another type of controller, such as
map-reduce or spark, may be a better fit.)
- Drawback: is a form of server-side templating, which we want in Kubernetes but
have not fully designed (see the [PetSets proposal](https://github.com/kubernetes/kubernetes/pull/18016/files?short_path=61f4179#diff-61f41798f4bced6e42e45731c1494cee)).

The index-only approach:

- Requires that the user keep the *per completion parameters* in a separate
storage, such as a configData or networked storage.
- Makes no changes to the JobSpec.
- Drawback: while in separate storage, they could be mutated, which would have
unexpected effects.
- Drawback: Logic for using index to lookup parameters needs to be in the Pod.
- Drawback: CLIs and UIs are limited to using the "index" as the identity of a
pod from a job. They cannot easily say, for example `repeated failures on the
pod processing banana.txt`.

Index-only approach relies on at least one of the following being true:

1. Image containing a shell and certain shell commands (not all images have
this).
1. Use directly consumes the index from annotations (file or env var) and
expands to specific behavior in the main program.

Also Using the index-only approach from non-kubectl clients requires that they
mimic the script-generation step, or only use the second style.

#### Decision

It is decided to implement the Index-only approach now. Once the server-side
templating design is complete for Kubernetes, and we have feedback from users,
we can consider if Multiple Substitution.

## Detailed Design

#### Job Resource Schema Changes

No changes are made to the JobSpec.


The JobStatus is also not changed. The user can gauge the progress of the job by
the `.status.succeeded` count.


#### Job Spec Compatilibity

A job spec written before this change will work exactly the same as before with
the new controller. The Pods it creates will have the same environment as
before. They will have a new annotation, but pod are expected to tolerate
unfamiliar annotations.

However, if the job controller version is reverted, to a version before this
change, the jobs whose pod specs depend on the new annotation will fail.
This is okay for a Beta resource.

#### Job Controller Changes

The Job controller will maintain for each Job a data structed which
indicates the status of each completion index. We call this the
*scoreboard* for short. It is an array of length `.spec.completions`.
Elements of the array are `enum` type with possible values including
`complete`, `running`, and `notStarted`.

The scoreboard is stored in Job Controller memory for efficiency. In either
case, the Status can be reconstructed from watching pods of the job (such as on
a controller manager restart). The index of the pods can be extracted from the
pod annotation.

When Job controller sees that the number of running pods is less than the
desired parallelism of the job, it finds the first index in the scoreboard with
value `notRunning`. It creates a pod with this creation index.

When it creates a pod with creation index `i`,  it makes a copy of the
`.spec.template`, and sets
`.spec.template.metadata.annotations.[kubernetes.io/job/completion-index]` to
`i`. It does this in both the index-only and multiple-substitutions options.

Then it creates the pod.

When the controller notices that a pod has completed or is running or failed,
it updates the scoreboard.

When all entries in the scoreboard are `complete`, then the job is complete.


#### Downward API Changes

The downward API is changed to support extracting specific key names into a
single environment variable. So, the following would be supported:

```
kind: Pod
version: v1
spec:
  containers:
  - name: foo
    env:
    - name: MY_INDEX
      valueFrom:
        fieldRef:
          fieldPath: metadata.annotations[kubernetes.io/job/completion-index]
```

This requires kubelet changes.

Users who fail to upgrade their kubelets at the same time as they upgrade their
controller manager will see a failure for pods to run when they are created by
the controller. The Kubelet will send an event about failure to create the pod.
The `kubectl describe job` will show many failed pods.


#### Kubectl Interface Changes

The `--completions` and `--completion-index-var-name` flags are added to
kubectl.

For example, this command:

```
kubectl run say-number --image=busybox \
   --completions=3 \
   --completion-index-var-name=I \
   -- \
   sh -c 'echo "My index is $I" && sleep 5' 
```

will run 3 pods to completion, each printing one of the following lines:

```
My index is 1
My index is 2
My index is 0
```

Kubectl would create the following pod:



Kubectl will also support the `--per-completion-env` flag, as described
previously. For example, this command:

```
kubectl run say-fruit --image=busybox \
   --per-completion-env=FRUIT="apple banana cherry" \
   --per-completion-env=COLOR="green yellow red" \
   -- \
   sh -c 'echo "Have a nice $COLOR $FRUIT" && sleep 5' 
```

or equivalently:

```
echo "apple banana cherry" > fruits.txt
echo "green yellow red" > colors.txt

kubectl run say-fruit --image=busybox \
   --per-completion-env=FRUIT="$(cat fruits.txt)" \
   --per-completion-env=COLOR="$(cat fruits.txt)" \
   -- \
   sh -c 'echo "Have a nice $COLOR $FRUIT" && sleep 5' 
```

or similarly:

```
kubectl run say-fruit --image=busybox \
   --per-completion-env=FRUIT=@fruits.txt \
   --per-completion-env=COLOR=@fruits.txt \
   -- \
   sh -c 'echo "Have a nice $COLOR $FRUIT" && sleep 5' 
```

will all run 3 pods in parallel. Index 0 pod will log:

```
Have a nice grenn apple
```

and so on.


Notes:

- `--per-completion-env=` is of form `KEY=VALUES` where `VALUES` is either a
quoted space separated list or `@` and the name of a text file containing a
list.
- `--per-completion-env=` can be specified several times, but all must have the
same length list.
- `--completions=N` with `N` equal to list length is implied.
- The flag `--completions=3` sets `job.spec.completions=3`.
- The flag `--completion-index-var-name=I` causes an env var to be created named
I in each pod, with the index in it.
- The flag `--restart=OnFailure` is implied by `--completions` or any
job-specific arguments. The user can also specify `--restart=Never` if they
desire but may not specify `--restart=Always` with job-related flags.
- Setting any of these flags in turn tells kubectl to create a Job, not a
replicationController.

#### How Kubectl Creates Job Specs.

To pass in the parameters, kubectl will generate a shell script which
can:
- parse the index from the annotation
- hold all the parameter lists.
- lookup the correct index in each parameter list and set an env var.

For example, consider this command:

```
kubectl run say-fruit --image=busybox \
   --per-completion-env=FRUIT="apple banana cherry" \
   --per-completion-env=COLOR="green yellow red" \
   -- \
   sh -c 'echo "Have a nice $COLOR $FRUIT" && sleep 5' 
```

First, kubectl generates the PodSpec as it normally does for `kubectl run`.

But, then it will generate this script:

```sh
#!/bin/sh
# Generated by kubectl run ...
# Check for needed commands
if [[ ! type cat ]]
then
  echo "$0: Image does not include required command: cat"
  exit 2
fi
if [[ ! type grep ]]
then
  echo "$0: Image does not include required command: grep"
  exit 2
fi
# Check that annotations are mounted from downward API
if [[ ! -e /etc/annotations ]]
then
  echo "$0: Cannot find /etc/annotations"
  exit 2
fi
# Get our index from annotations file
I=$(cat /etc/annotations | grep job.kubernetes.io/index | cut -f 2 -d '\"') || echo "$0: failed to extract index"
export I

# Our parameter lists are stored inline in this script.
FRUIT_0="apple"
FRUIT_1="banana"
FRUIT_2="cherry"
# Extract the right parameter value based on our index.
# This works on any Bourne-based shell.
FRUIT=$(eval echo \$"FRUIT_$I")
export FRUIT

COLOR_0="green"
COLOR_1="yellow"
COLOR_2="red"

COLOR=$(eval echo \$"FRUIT_$I")
export COLOR
```

Then it POSTs this script, encoded, inside a ConfigData.
It attaches this volume to the PodSpec.

Then it will edit the command line of the Pod to run this script before the rest of
the command line.

Then it appends a DownwardAPI volume to the pod spec to get the annotations in a file, like this:
It also appends the Secret (later configData) volume with the script in it.

So, the Pod template that kubectl creates (inside the job template) looks like this:

```
apiVersion: v1
kind: Job
...
spec:
  ...
  template:
    ...
    spec:
      containers:
        - name: c
          image: gcr.io/google_containers/busybox
          command:
            - 'sh'
            - '-c'
            - '/etc/job-params.sh; echo "this is the rest of the command"'
          volumeMounts:
            - name: annotations
              mountPath: /etc 
            - name: script
              mountPath: /etc
      volumes:
        - name: annotations
          downwardAPI:
            items:
              - path: "annotations"
                ieldRef:
                  fieldPath: metadata.annotations
        - name: script
          secret:
            secretName: jobparams-abc123
```

###### Alternatives

Kubectl could append a `valueFrom` line like this to
get the index into the environment:

```yaml
apiVersion: extensions/v1beta1
kind: Job
metadata:
  ...
spec:
  ...
  template:
    ...
    spec:
      containers:
      - name: foo 
        ...
        env:        
 # following block added:
          - name: I
            valueFrom:
             fieldRef:
               fieldPath:  metadata.annotations."kubernetes.io/job-idx"
```

However, in order to inject other env vars from parameter list,
kubectl still needs to edit the command line.

Parameter lists could be passed via a configData volume instead of a secret.
Kubectl can be changed to work that way once the configData implementation is
complete.

Parameter lists could be passed inside an EnvVar.  This would have length
limitations, would pollute the output of `kubectl describe pods` and `kubectl
get pods -o json`.

Parameter lists could be passed inside an annotation.  This would have length
limitations, would pollute the output of `kubectl describe pods` and `kubectl
get pods -o json`.  Also, currently annotations can only be extracted into a
single file.  Complex logic is then needed to filter out exactly the desired
annotation data.

Bash array variables could simplify extraction of a particular parameter from a
list of parameters.  However, some popular base images do not include
`/bin/bash`.  For example, `busybox` uses a compact `/bin/sh` implementation
that does not support array syntax.

Kubelet does support [expanding variables without a
shell](http://kubernetes.io/kubernetes/v1.1/docs/design/expansion.html).  But it does not
allow for recursive substitution, which is required to extract the correct
parameter from a list based on the completion index of the pod.  The syntax
could be extended, but doing so seems complex and will be an unfamiliar syntax
for users.

Putting all the command line editing into a script and running that causes
the least pollution to the original command line, and it allows
for complex error handling.

Kubectl could store the script in an [Inline Volume](
https://github.com/kubernetes/kubernetes/issues/13610) if that proposal
is approved. That would remove the need to manage the lifetime of the
configData/secret, and prevent the case where someone changes the
configData mid-job, and breaks things in a hard-to-debug way.


## Interactions with other features

#### Supporting Work Queue Jobs too

For Work Queue Jobs, completions has no meaning. Parallelism should be allowed
to be greater than it, and pods have no identity. So, the job controller should
not create a scoreboard in the JobStatus, just a count.  Therefore, we need to
add one of the following to JobSpec:

- allow unset `.spec.completions` to indicate no scoreboard, and no index for
tasks (identical tasks).
- allow `.spec.completions=-1` to indicate the same.
- add `.spec.indexed` to job to indicate need for scoreboard.

#### Interaction with vertical autoscaling

Since pods of the same job will not be created with different resources,
a vertical autoscaler will need to:

- if it has index-specific initial resource suggestions, suggest those at
admission time; it will need to understand indexes.
- mutate resource requests on already created pods based on usage trend or
previous container failures.
- modify the job template, affecting all indexes.

#### Comparison to PetSets

The *Index substitution-only* option corresponds roughly to PetSet Proposal 1b.
The `perCompletionArgs` approach is similar to PetSet Proposal 1e, but more
restrictive and thus less verbose.

It would be easier for users if Indexed Job and PetSet are similar where
possible. However, PetSet differs in several key respects:

- PetSet is for ones to tens of instances.  Indexed job should work with tens of
thousands of instances.
- When you have few instances, you may want to given them pet names. When you
have many instances, you that many instances, integer indexes make more sense.
- When you have thousands of instances, storing the work-list in the JobSpec
is verbose.  For PetSet, this is less of a problem.
- PetSets (apparently) need to differ in more fields than indexed Jobs.

This differs from PetSet in that PetSet uses names and not indexes. PetSet is
intended to support ones to tens of things.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/indexed-job.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
