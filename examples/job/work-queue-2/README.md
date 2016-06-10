<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Example: Job with Work Queue with Pod Per Work Item

In this example, we will run a Kubernetes Job with multiple parallel
worker processes.  You may want to be familiar with the basic,
non-parallel, use of [Job](../../../docs/user-guide/jobs.md) first.

In this example, as each pod is created, it picks up one unit of work
from a task queue, completes it, deletes it from the queue, and exits.


Here is an overview of the steps in this example:

1. **Start a storage service to hold the work queue.**  In this example, we use Redis to store
  our work items.  In the previous example, we used RabbitMQ.  In this example, we use Redis and
  a custom work-queue client library because AMQP does not provide a good way for clients to
  detect when a finite-length work queue is empty.  In practice you would set up a store such
  as Redis once and reuse it for the work queues of many jobs, and other things.
1. **Create a queue, and fill it with messages.**  Each message represents one task to be done.  In
  this example, a message is just an integer that we will do a lengthy computation on.
1. **Start a Job that works on tasks from the queue**.  The Job starts several pods.  Each pod takes
  one task from the message queue, processes it, and repeats until the end of the queue is reached.


## Starting Redis

For this example, for simplicitly, we will start a single instance of Redis.
See the [Redis Example](../../../examples/redis/README.md) for an example
of deploying Redis scaleably and redundantly.

Start a temporary Pod running Redis and a service so we can find it.

```console
$ kubectl create -f examples/job/work-queue-2/redis-pod.yaml
pod "redis-master" created
$ kubectl create -f examples/job/work-queue-2/redis-service.yaml
service "redis" created
```

## Filling the Queue with tasks

Now lets fill the queue with some "tasks".  In our example, our tasks are just strings to be
printed.

Start a temporary interactive pod for running the Redis CLI

```console
$ kubectl run -i --tty temp --image redis --command "/bin/sh"
Waiting for pod default/redis2-c7h78 to be running, status is Pending, pod ready: false
Hit enter for command prompt
```

Now hit enter, start the redis CLI, and create a list with some work items in it.

```
# redis-cli -h redis
redis:6379> rpush job2 "apple"
(integer) 1
redis:6379> rpush job2 "banana"
(integer) 2
redis:6379> rpush job2 "cherry"
(integer) 3
redis:6379> rpush job2 "date"
(integer) 4
redis:6379> rpush job2 "fig"
(integer) 5
redis:6379> rpush job2 "grape"
(integer) 6
redis:6379> rpush job2 "lemon"
(integer) 7
redis:6379> rpush job2 "melon"
(integer) 8
redis:6379> rpush job2 "orange"
(integer) 9
redis:6379> lrange job2 0 -1
1) "apple"
2) "banana"
3) "cherry"
4) "date"
5) "fig"
6) "grape"
7) "lemon"
8) "melon"
9) "orange"
```

So, the list with key `job2` will be our work queue.

Note: if you do not have Kube DNS setup correctly, you may need to change
the first step of the above block to `redis-cli -h $REDIS_SERVICE_HOST`.


## Create an Image

Now we are ready to create an image that we will run.

We will use a python worker program with a redis client to read
the messages from the message queue.

A simple Redis work queue client library is provided,
called rediswq.py ([Download](rediswq.py?raw=true)).

The "worker" program in each Pod of the Job uses the work queue
client library to get work.  Here it is:

<!-- BEGIN MUNGE: EXAMPLE worker.py -->

```
#!/usr/bin/env python

# Copyright 2015 The Kubernetes Authors All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import rediswq

host="redis"
# Uncomment next two lines if you do not have Kube-DNS working.
# import os
# host = os.getenv("REDIS_SERVICE_HOST")

q = rediswq.RedisWQ(name="job2", host="redis")
print("Worker with sessionID: " +  q.sessionID())
print("Initial queue state: empty=" + str(q.empty()))
while not q.empty():
  item = q.lease(lease_secs=10, block=True, timeout=2) 
  if item is not None:
    itemstr = item.decode("utf=8")
    print("Working on " + itemstr)
    time.sleep(10) # Put your actual work here instead of sleep.
    q.complete(item)
  else:
    print("Waiting for work")
print("Queue empty, exiting")
```

[Download example](worker.py?raw=true)
<!-- END MUNGE: EXAMPLE worker.py -->

If you are working from the source tree,
change directory to the `examples/job/work-queue-2` directory.
Otherwise, download [`worker.py`](worker.py?raw=true), [`rediswq.py`](rediswq.py?raw=true), and [`Dockerfile`](Dockerfile?raw=true)
using above links. Then build the image:

```console
$ docker build -t job-wq-2 .
```

### Push the image

For the [Docker Hub](https://hub.docker.com/), tag your app image with
your username and push to the Hub with the below commands. Replace
`<username>` with your Hub username.

```
docker tag job-wq-2 <username>/job-wq-2
docker push <username>/job-wq-2
```

You need to push to a public repository or [configure your cluster to be able to access
your private repository](../../../docs/user-guide/images.md).

If you are using [Google Container
Registry](https://cloud.google.com/tools/container-registry/), tag
your app image with your project ID, and push to GCR. Replace
`<project>` with your project ID.

```
docker tag job-wq-2 gcr.io/<project>/job-wq-2
gcloud docker push gcr.io/<project>/job-wq-2
```

## Defining a Job

Here is the job definition:


<!-- BEGIN MUNGE: EXAMPLE job.yaml -->

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: job-wq-2
spec:
  parallelism: 2
  template:
    metadata:
      name: job-wq-2
    spec:
      containers:
      - name: c
        image: gcr.io/myproject/job-wq-2
      restartPolicy: OnFailure
```

[Download example](job.yaml?raw=true)
<!-- END MUNGE: EXAMPLE job.yaml -->

Be sure to edit the job template to
change `gcr.io/myproject` to your own path.

In this example, each pod works on several items from the queue and then exits when there are no more items.
Since the workers themselves detect when the workqueue is empty, and the Job controller does not
know about the workqueue, it relies on the workers to signal when they are done working.
The workers signal that the queue is empty by exiting with success.  So, as soon as any worker
exits with success, the controller knows the work is done, and the Pods will exit soon.
So, we set the completion count of the Job to 1.  The job controller will wait for the other pods to complete
too.


## Running the Job

So, now run the Job:

```console
$ kubectl create -f ./job.yaml
```

Now wait a bit, then check on the job.

```console
$ ./kubectl describe jobs/job-wq-2 
Name:		job-wq-2
Namespace:	default
Image(s):	gcr.io/exampleproject/job-wq-2
Selector:	app in (job-wq-2)
Parallelism:	2
Completions:	Unset
Start Time:	Mon, 11 Jan 2016 17:07:59 -0800
Labels:		app=job-wq-2
Pods Statuses:	1 Running / 0 Succeeded / 0 Failed
No volumes.
Events:
  FirstSeen	LastSeen	Count	From			SubobjectPath	Type		Reason			Message
  ---------	--------	-----	----			-------------	--------	------			-------
  33s		33s		1	{job-controller }			Normal		SuccessfulCreate	Created pod: job-wq-2-lglf8


$ kubectl logs pods/job-wq-2-7r7b2
Worker with sessionID: bbd72d0a-9e5c-4dd6-abf6-416cc267991f
Initial queue state: empty=False
Working on banana
Working on date
Working on lemon
```

As you can see, one of our pods worked on several work units.

## Alternatives

If running a queue service or modifying your containers to use a work queue is inconvenient, you may
want to consider one of the other [job patterns](../../../docs/user-guide/jobs.md#job-patterns).

If you have a continuous stream of background processing work to run, then
consider running your background workers with a `replicationController` instead,
and consider running a background processing library such as
https://github.com/resque/resque.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/job/work-queue-2/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
