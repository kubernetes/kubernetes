

!Disambiguation of plugin types

1. Value/setting plugin.
 a. Example: You've implemented the ultimate persistent storage solution, FrobStore. You want to mount volumes sourced from it inside your containers.
 b. You start passing FrobVolumeSources in your pods.
 c. You write a little bit of code that knows how to mount a FrobVolume (could be a shell script, could be something more complicated).
 d. You install a handler in Kubelet that calls your code when a FrobVolumeSource is encountered. (TODO!)

2. Component plugin.
 a. Example: You want a ShardController to run your pods. It devides up some key space and sets a label (e.g., "shardRegexp=[a-b].*") on each pod.
 b. You write the controller code-- it's almost a copy of ReplicationController.
 c. You package it up in a docker image.
 d. You're going to run your shardController from a replicationController. So you make a .json file that starts a replicationController that starts your shardController, and start it.
 e. You make a service.json to direct traffic to your running shardControllers, and start it.
 f. You make a componentPlugin.json file, and start it. This causes traffic to <apiserver's ip address>/api/v1beta1/shardController to be redirected to your service.

!This doc is about component plugins.

A helpful analogy: If Kubernetes is defining a "CloudOS" and the core components form a kernel, then component plugins could be thought of as "cloud device drivers".

Q. Why do we need this concept at all?
A. Multiple answers:
 a. "microkernel" architecture. Keeping the core kubernetes components as simple as possible should make it easier to make/keep them bug-free and performant.
 a. Different kubernetes api endpoints may need to scale separately.
 a. A single plugin could fail without bringing the whole cluster down.
 a. There are some cases where people can't, or won't, share code. For these cases, separate binaries need to be able to work together.
 a. (Minor) Security is much easier to reason about if you're not compiling arbitrary code into your kubernetes apiserver.

Q. This makes the system hard to reason about.
A. Well, it makes each piece easy to reason about. When reasoning about the system as a whole, one should look at our (future) config setup. Clusters being complex is going to happen no matter what, we might as well do it with understandable pieces.

Q. Why run a component plugin from a replication controller?
A. This allows scaling and ensures that your plugin will be restarted if it crashes or the minion it's on dies, etc..

Q. Why require a service? Why not just an arbitrary endpoint?
A. If you're running as a k8s component behind a replication controller, there's really no other way to identify your current location, at least not currently. We might reevaluate this when IP-per-service is implemented.

Q. What if a plugin wants to handle multiple api endpoints?
A. Put multiple ComponentPlugin objects into the system.

Q. How do component plugins store their stuff?
A. Choices:
 a. We could make them just deal with it.
 b. We could allow them to delegate storage to our apiserver. E.g., we allow the plugin (but no one else) to use the path "/api/v1beta1/delegatedStorage/<pluginID>[/objectname]" to POST/PUT/GET/DELETE and watch objects.
 c. We could be even more helpful (see below)

Q. How do REST requests that involve a plugin work?
A. Choices:
 a. Plugin as completely contained entity:
  1. User/system component sends request to /api/v1beta1/shardController
  2. API server either 307 redirects or proxies to <shardControllerLocation>/api/v1beta1/shardController
  3. The shardController performs some action and returns.
   i. Oh the shardController needs to store some data?
    a. It turns around and POSTs to /api/v1beta1/delegatedStorage/shardController or PUTs to /api/v1beta1/delegatedStorage/shardController/ID
    b. Or: it stores data in its own system.
  4. User is happy.

 b. Plugin as webhooks:
  1. User/system component sends request to /api/v1beta1/shardController
   i. POST:
    a. apiserver POSTs unaltered user data to /api/v1beta1/delegatedStorage/shardController.
    b. shardController does any tweaking to the data, and returns either validated data with a 200 or some 4xx error.
    c. apiserver stores this validated data.
    d. apiserver returns stored data to client.
    e. Might have to do a-c in a loop to be race-free.
   ii. PUT:
    a. apiserver PUTs unaltered user data to /api/v1beta1/delegatedStorage/shardController/ID.
    b. Oh actually that doesn't work, we need to send both current state and user's PUT data.
    c. Oh and do that in an etcd compare-and-swap loop because otherwise there will be races.
    d. Webhooks are complex and hard.
    e. Let's not do webhooks.
   iii. GET & list:
    a. apiserver reads object(s) from etcd and returns them. No webhook!
   iv. DELETE:
    a. apiserver sends DELETE to /api/v1beta1/delegatedStorage/shardController/ID; if it returns a 200, apiserver deletes the object.
   v. watches:
    a. apiservers sets up a watch. Label queries work, field queries TBD.

