<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Example: Multiple Job Objects from Template Expansion

In this example, we will run multiple Kubernetes Jobs created from
a common template.  You may want to be familiar with the basic,
non-parallel, use of [Job](../../../docs/user-guide/jobs.md) first.

## Basic Template Expansion

First, create a template of a Job object:

<!-- BEGIN MUNGE: EXAMPLE job.yaml.txt -->

```
apiVersion: extensions/v1beta1
kind: Job
metadata:
  name: process-item-$ITEM
spec:
  selector:
    matchLabels:
      app: jobexample
      item: $ITEM
  template:
    metadata:
      name: jobexample
      labels:
        app: jobexample
        item: $ITEM
    spec:
      containers:
      - name: c
        image: busybox
        command: ["sh", "-c", "echo Processing item $ITEM && sleep 5"]
      restartPolicy: Never
```

[Download example](job.yaml.txt?raw=true)
<!-- END MUNGE: EXAMPLE job.yaml.txt -->

Unlike a *pod template*, our *job template* is not a Kubernetes API type.  It is just
a yaml representation of a Job object that has some placeholders that need to be filled
in before it can be used.  The `$ITEM` syntax is not meaningful to Kubernetes.

In this example, the only processing the container does is to `echo` a string and sleep for a bit.
In a real use case, the processing would be some substantial computation, such as rendering a frame
of a movie, or processing a range of rows in a database.  The "$ITEM" parameter would specify for
example, the frame number or the row range.

This Job has two labels. The first label, `app=jobexample`, distinguishes this group of jobs from
other groups of jobs (these are not shown, but there might be other ones).  This label
makes it convenient to operate on all the jobs in the group at once.  The second label, with
key `item`, distinguishes individual jobs in the group.  Each Job object needs to have
a unique label that no other job has.  This is it.
Neither of these label keys are special to kubernetes -- you can pick your own label scheme.

Next, expand the template into multiple files, one for each item to be processed.

```console
# Expand files into a temporary directory
$ mkdir ./jobs
$ for i in apple banana cherry
do
  cat job.yaml.txt | sed "s/\$ITEM/$i/" > ./jobs/job-$i.yaml
done
$ ls jobs/
job-apple.yaml
job-banana.yaml
job-cherry.yaml
```

Here, we used `sed` to replace the string `$ITEM` with the the loop variable.
You could use any type of template language (jinja2, erb) or write a program
to generate the Job objects.

Next, create all the jobs with one kubectl command:

```console
$ kubectl create -f ./jobs
job "process-item-apple" created
job "process-item-banana" created
job "process-item-cherry" created
```

Now, check on the jobs:

```console
$ kubectl get jobs -l app=jobexample -L item
JOB                   CONTAINER(S)   IMAGE(S)   SELECTOR                               SUCCESSFUL ITEM
process-item-apple    c              busybox    app in (jobexample),item in (apple)    1          apple
process-item-banana   c              busybox    app in (jobexample),item in (banana)   1          banana
process-item-cherry   c              busybox    app in (jobexample),item in (cherry)   1          cherry
```

Here we use the `-l` option to select all jobs that are part of this
group of jobs.  (There might be other unrelated jobs in the system that we
do not care to see.)

The `-L` option adds an extra column with just the `item` label value.

We can check on the pods as well using the same label selector:

```console
$ kubectl get pods -l app=jobexample -L item
NAME                        READY     STATUS      RESTARTS   AGE       ITEM
process-item-apple-kixwv    0/1       Completed   0          4m        apple
process-item-banana-wrsf7   0/1       Completed   0          4m        banana
process-item-cherry-dnfu9   0/1       Completed   0          4m        cherry
```

There is not a single command to check on the output of all jobs at once,
but looping over all the pods is pretty easy:

```console
$ for p in $(kubectl get pods -l app=jobexample -o name)
do
  kubectl logs $p
done
Processing item apple
Processing item banana
Processing item cherry
```

## Multiple Template Parameters

In the first example, each instance of the template had one parameter, and that parameter was also
used as a label.  However label keys are limited in [what characters they can
contain](labels.md#syntax-and-character-set).

This slightly more complex example uses a the jinja2 template language to generate our objects.
We will use a one-line python script to convert the template to a file.

First, download or paste the following template file to a file called `job.yaml.jinja2`:

<!-- BEGIN MUNGE: EXAMPLE job.yaml.jinja2 -->

```
{%- set params = [{ "name": "apple", "url": "http://www.orangepippin.com/apples", },
                  { "name": "banana", "url": "https://en.wikipedia.org/wiki/Banana", },
                  { "name": "raspberry", "url": "https://www.raspberrypi.org/" }]
%}
{%- for p in params %}
{%- set name = p["name"] %}
{%- set url = p["url"] %}
apiVersion: extensions/v1beta1
kind: Job
metadata:
  name: jobexample-{{ name }}
spec:
  selector:
    matchLabels:
      app: jobexample
      item: {{ name }}
  template:
    metadata:
      name: jobexample
      labels:
        app: jobexample
        item: {{ name }}
    spec:
      containers:
      - name: c
        image: busybox
        command: ["sh", "-c", "echo Processing URL {{ url }} && sleep 5"]
      restartPolicy: Never
---
{%- endfor %}
```

[Download example](job.yaml.jinja2?raw=true)
<!-- END MUNGE: EXAMPLE job.yaml.jinja2 -->

The above template defines parameters for each job object using a list of
python dicts (lines 1-4).  Then a for loop emits one job yaml object
for each set of parameters (remaining lines).
We take advantage of the fact that multiple yaml documents can be concatenated
with the `---` separator (second to last line).
.)  We can pipe the output directly to kubectl to
create the objects.

You will need the jinja2 package if you do not already have it: `pip install --user jinja2`.
Now, use this one-line python program to expand the template:

```
$ alias render_template='python -c "from jinja2 import Template; import sys; print(Template(sys.stdin.read()).render());"'
```



The output can be saved to a file, like this:

```
$ cat job.yaml.jinja2 | render_template > jobs.yaml
```

or sent directly to kubectl, like this:

```
$ cat job.yaml.jinja2 | render_template | kubectl create -f -
```

## Alternatives

If you have a large number of job objects, you may find that:

- even using labels, managing so many Job objects is cumbersome.
- you exceed resource quota when creating all the Jobs at once,
  and do not want to wait to create them incrementally.
- you need a way to easily scale the number of pods running
  concurrently.  One reason would be to avoid using too many
  compute resources.  Another would be to limit the number of
  concurrent requests to a shared resource, such as a database,
  used by all the pods in the job.
- very large numbers of jobs created at once overload the
  kubernetes apiserver, controller, or scheduler.

In this case, you can consider one of the
other [job patterns](../../../docs/user-guide/jobs.md#job-patterns).


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/job/expansions/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
