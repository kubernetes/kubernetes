# Soak Test serve_hostnames
This directory contains the source for a soak test `serve_hostnames` which performs the following actions when used with the GCE provider:

* A connection is established to the master of the cluster identified from the current context set in `$HOME/.kube/.kubeconfig`.
* The nodes available on the cluster are enumerated (say *N* nodes).
* On each node, *M* pods are created (by default 1). The pod encapsulates the `serve_hostnames` image which simply returns the name of the pod in response to a `GET` request.
The pods are created individually (i.e. not with a replication controller).
* A service is created which maps to these pods.
* The program makes *I* iterations (default 1) where it issues *QxNxM* queries (*Q* default is 10) via the service proxy interface at the master.
* The program verifies that every pod (and thus every node) responded to at least one query (the average should be about *Q*).
* The time taken to perform various operations is reported and some operations are re-tried if they failed.

Run the following command:
```sh
./serve_hostnames
```

Here is some representative output:
```
I0326 14:21:04.179893   11434 serve_hostnames.go:60] Starting serve_hostnames soak test with queries=10 and podsPerNode=1 upTo=1
I0326 14:21:04.507252   11434 serve_hostnames.go:85] Nodes found on this cluster:
I0326 14:21:04.507282   11434 serve_hostnames.go:87] 0: kubernetes-node-5h4m.c.kubernetes-satnam.internal
I0326 14:21:04.507297   11434 serve_hostnames.go:87] 1: kubernetes-node-9i4n.c.kubernetes-satnam.internal
I0326 14:21:04.507309   11434 serve_hostnames.go:87] 2: kubernetes-node-d0yo.c.kubernetes-satnam.internal
I0326 14:21:04.507320   11434 serve_hostnames.go:87] 3: kubernetes-node-jay1.c.kubernetes-satnam.internal
I0326 14:21:04.507347   11434 serve_hostnames.go:95] Using namespace serve-hostnames-8145 for this test.
I0326 14:21:04.507363   11434 serve_hostnames.go:98] Creating service serve-hostnames-8145/serve-hostnames
I0326 14:21:04.559849   11434 serve_hostnames.go:148] Creating pod serve-hostnames-8145/serve-hostname-0-0 on node kubernetes-node-5h4m.c.kubernetes-satnam.internal
I0326 14:21:04.605603   11434 serve_hostnames.go:148] Creating pod serve-hostnames-8145/serve-hostname-1-0 on node kubernetes-node-9i4n.c.kubernetes-satnam.internal
I0326 14:21:04.662099   11434 serve_hostnames.go:148] Creating pod serve-hostnames-8145/serve-hostname-2-0 on node kubernetes-node-d0yo.c.kubernetes-satnam.internal
I0326 14:21:04.707179   11434 serve_hostnames.go:148] Creating pod serve-hostnames-8145/serve-hostname-3-0 on node kubernetes-node-jay1.c.kubernetes-satnam.internal
I0326 14:21:04.757646   11434 serve_hostnames.go:194] Waiting for the serve-hostname pods to be ready
I0326 14:23:31.125188   11434 serve_hostnames.go:211] serve-hostnames-8145/serve-hostname-0-0 is running
I0326 14:23:31.165984   11434 serve_hostnames.go:211] serve-hostnames-8145/serve-hostname-1-0 is running
I0326 14:25:22.213751   11434 serve_hostnames.go:211] serve-hostnames-8145/serve-hostname-2-0 is running
I0326 14:25:37.387257   11434 serve_hostnames.go:211] serve-hostnames-8145/serve-hostname-3-0 is running
W0326 14:25:39.243813   11434 serve_hostnames.go:265] No response from pod serve-hostname-3-0 on node kubernetes-node-jay1.c.kubernetes-satnam.internal at iteration 0
I0326 14:25:39.243844   11434 serve_hostnames.go:269] Iteration 0 took 1.814483599s for 40 queries (22.04 QPS)
I0326 14:25:39.243871   11434 serve_hostnames.go:182] Cleaning up pods
I0326 14:25:39.434619   11434 serve_hostnames.go:130] Cleaning up service serve-hostnames-8145/server-hostnames
```

The pods are named with -*N*-*M* suffixes which identify the number of the node *N* and the number of the pod *M* on that node.
Notice that in this run the pod (number 0) running on node 3 did not respond to any queries.

The number of iterations to perform for issuing queries can be changed from the default of 1 to some higher value e.g. `--up_to=3` and the number of pods per node can also be changed e.g. `--pods_per_node=2`:

```sh
./serve_hostnames --up_to=3 --pods_per_node=2
```

The output is similar to this:
```
I0326 14:27:27.584378   11808 serve_hostnames.go:60] Starting serve_hostnames soak test with queries=10 and podsPerNode=2 upTo=3
I0326 14:27:27.913713   11808 serve_hostnames.go:85] Nodes found on this cluster:
I0326 14:27:27.913774   11808 serve_hostnames.go:87] 0: kubernetes-node-5h4m.c.kubernetes-satnam.internal
I0326 14:27:27.913800   11808 serve_hostnames.go:87] 1: kubernetes-node-9i4n.c.kubernetes-satnam.internal
I0326 14:27:27.913825   11808 serve_hostnames.go:87] 2: kubernetes-node-d0yo.c.kubernetes-satnam.internal
I0326 14:27:27.913846   11808 serve_hostnames.go:87] 3: kubernetes-node-jay1.c.kubernetes-satnam.internal
I0326 14:27:27.913904   11808 serve_hostnames.go:95] Using namespace serve-hostnames-4997 for this test.
I0326 14:27:27.913931   11808 serve_hostnames.go:98] Creating service serve-hostnames-4997/serve-hostnames
I0326 14:27:27.969083   11808 serve_hostnames.go:148] Creating pod serve-hostnames-4997/serve-hostname-0-0 on node kubernetes-node-5h4m.c.kubernetes-satnam.internal
I0326 14:27:28.020133   11808 serve_hostnames.go:148] Creating pod serve-hostnames-4997/serve-hostname-0-1 on node kubernetes-node-5h4m.c.kubernetes-satnam.internal
I0326 14:27:28.070054   11808 serve_hostnames.go:148] Creating pod serve-hostnames-4997/serve-hostname-1-0 on node kubernetes-node-9i4n.c.kubernetes-satnam.internal
I0326 14:27:28.118641   11808 serve_hostnames.go:148] Creating pod serve-hostnames-4997/serve-hostname-1-1 on node kubernetes-node-9i4n.c.kubernetes-satnam.internal
I0326 14:27:28.168786   11808 serve_hostnames.go:148] Creating pod serve-hostnames-4997/serve-hostname-2-0 on node kubernetes-node-d0yo.c.kubernetes-satnam.internal
I0326 14:27:28.214730   11808 serve_hostnames.go:148] Creating pod serve-hostnames-4997/serve-hostname-2-1 on node kubernetes-node-d0yo.c.kubernetes-satnam.internal
I0326 14:27:28.261685   11808 serve_hostnames.go:148] Creating pod serve-hostnames-4997/serve-hostname-3-0 on node kubernetes-node-jay1.c.kubernetes-satnam.internal
I0326 14:27:28.320224   11808 serve_hostnames.go:148] Creating pod serve-hostnames-4997/serve-hostname-3-1 on node kubernetes-node-jay1.c.kubernetes-satnam.internal
I0326 14:27:28.387007   11808 serve_hostnames.go:194] Waiting for the serve-hostname pods to be ready
I0326 14:28:28.969149   11808 serve_hostnames.go:211] serve-hostnames-4997/serve-hostname-0-0 is running
I0326 14:28:29.010376   11808 serve_hostnames.go:211] serve-hostnames-4997/serve-hostname-0-1 is running
I0326 14:28:29.050463   11808 serve_hostnames.go:211] serve-hostnames-4997/serve-hostname-1-0 is running
I0326 14:28:29.091164   11808 serve_hostnames.go:211] serve-hostnames-4997/serve-hostname-1-1 is running
I0326 14:30:00.850461   11808 serve_hostnames.go:211] serve-hostnames-4997/serve-hostname-2-0 is running
I0326 14:30:00.891559   11808 serve_hostnames.go:211] serve-hostnames-4997/serve-hostname-2-1 is running
I0326 14:30:00.932829   11808 serve_hostnames.go:211] serve-hostnames-4997/serve-hostname-3-0 is running
I0326 14:30:00.973941   11808 serve_hostnames.go:211] serve-hostnames-4997/serve-hostname-3-1 is running
W0326 14:30:04.726582   11808 serve_hostnames.go:265] No response from pod serve-hostname-2-0 on node kubernetes-node-d0yo.c.kubernetes-satnam.internal at iteration 0
W0326 14:30:04.726658   11808 serve_hostnames.go:265] No response from pod serve-hostname-2-1 on node kubernetes-node-d0yo.c.kubernetes-satnam.internal at iteration 0
I0326 14:30:04.726696   11808 serve_hostnames.go:269] Iteration 0 took 3.711080213s for 80 queries (21.56 QPS)
W0326 14:30:08.267297   11808 serve_hostnames.go:265] No response from pod serve-hostname-2-0 on node kubernetes-node-d0yo.c.kubernetes-satnam.internal at iteration 1
W0326 14:30:08.267365   11808 serve_hostnames.go:265] No response from pod serve-hostname-2-1 on node kubernetes-node-d0yo.c.kubernetes-satnam.internal at iteration 1
I0326 14:30:08.267404   11808 serve_hostnames.go:269] Iteration 1 took 3.540635303s for 80 queries (22.59 QPS)
I0326 14:30:11.971349   11808 serve_hostnames.go:269] Iteration 2 took 3.703884372s for 80 queries (21.60 QPS)
I0326 14:30:11.971425   11808 serve_hostnames.go:182] Cleaning up pods
I0326 14:30:12.382932   11808 serve_hostnames.go:130] Cleaning up service serve-hostnames-4997/server-hostnames
```

Notice here that for the first two iterations neither of the pods on node 2 responded but by the third iteration responses
were received from all nodes.

For a soak test use `--up_to=-1` which will loop indefinitely.


Note that this is not designed to be a performance test. The goal for this program is to provide an easy way to have a soak test
that can run indefinitely an exercise enough of Kubernetes' functionality to be confident that the cluster is still up and healthy.
The reported QPS mainly indicates latency to the master since the proxy requests are issued (deliberately) in a serial manner.


A more detailed report can be produced with `--v=4` which measures the time taken to perform various operations
and it also reports the distribution of responses received from the pods. In the example below
we see that the pod on node 0 returned 18 responses, the pod on node 1 returned 10 responses and the
pod on node 3 returned 12 responses and the pod on node 2 did not respond at all.

```sh
./serve_hostnames --v=4
```

The output is similar to this:
```
I0326 14:33:26.020917   12099 serve_hostnames.go:60] Starting serve_hostnames soak test with queries=10 and podsPerNode=1 upTo=1
I0326 14:33:26.365201   12099 serve_hostnames.go:85] Nodes found on this cluster:
I0326 14:33:26.365260   12099 serve_hostnames.go:87] 0: kubernetes-node-5h4m.c.kubernetes-satnam.internal
I0326 14:33:26.365288   12099 serve_hostnames.go:87] 1: kubernetes-node-9i4n.c.kubernetes-satnam.internal
I0326 14:33:26.365313   12099 serve_hostnames.go:87] 2: kubernetes-node-d0yo.c.kubernetes-satnam.internal
I0326 14:33:26.365334   12099 serve_hostnames.go:87] 3: kubernetes-node-jay1.c.kubernetes-satnam.internal
I0326 14:33:26.365392   12099 serve_hostnames.go:95] Using namespace serve-hostnames-1631 for this test.
I0326 14:33:26.365419   12099 serve_hostnames.go:98] Creating service serve-hostnames-1631/serve-hostnames
I0326 14:33:26.423927   12099 serve_hostnames.go:118] Service create serve-hostnames-1631/server-hostnames took 58.473361ms
I0326 14:33:26.423981   12099 serve_hostnames.go:148] Creating pod serve-hostnames-1631/serve-hostname-0-0 on node kubernetes-node-5h4m.c.kubernetes-satnam.internal
I0326 14:33:26.480185   12099 serve_hostnames.go:168] Pod create serve-hostnames-1631/serve-hostname-0-0 request took 56.178906ms
I0326 14:33:26.480271   12099 serve_hostnames.go:148] Creating pod serve-hostnames-1631/serve-hostname-1-0 on node kubernetes-node-9i4n.c.kubernetes-satnam.internal
I0326 14:33:26.534300   12099 serve_hostnames.go:168] Pod create serve-hostnames-1631/serve-hostname-1-0 request took 53.981761ms
I0326 14:33:26.534396   12099 serve_hostnames.go:148] Creating pod serve-hostnames-1631/serve-hostname-2-0 on node kubernetes-node-d0yo.c.kubernetes-satnam.internal
I0326 14:33:26.590188   12099 serve_hostnames.go:168] Pod create serve-hostnames-1631/serve-hostname-2-0 request took 55.752115ms
I0326 14:33:26.590222   12099 serve_hostnames.go:148] Creating pod serve-hostnames-1631/serve-hostname-3-0 on node kubernetes-node-jay1.c.kubernetes-satnam.internal
I0326 14:33:26.650024   12099 serve_hostnames.go:168] Pod create serve-hostnames-1631/serve-hostname-3-0 request took 59.781614ms
I0326 14:33:26.650083   12099 serve_hostnames.go:194] Waiting for the serve-hostname pods to be ready
I0326 14:33:32.776651   12099 serve_hostnames.go:211] serve-hostnames-1631/serve-hostname-0-0 is running
I0326 14:33:32.822324   12099 serve_hostnames.go:211] serve-hostnames-1631/serve-hostname-1-0 is running
I0326 14:35:03.741235   12099 serve_hostnames.go:211] serve-hostnames-1631/serve-hostname-2-0 is running
I0326 14:35:03.786411   12099 serve_hostnames.go:211] serve-hostnames-1631/serve-hostname-3-0 is running
I0326 14:35:03.878030   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 45.656425ms
I0326 14:35:03.923999   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 45.887564ms
I0326 14:35:03.967731   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.7004ms
I0326 14:35:04.011077   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.318018ms
I0326 14:35:04.054958   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.843043ms
I0326 14:35:04.099051   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 44.030505ms
I0326 14:35:04.143197   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 44.069434ms
I0326 14:35:04.186800   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.530301ms
I0326 14:35:04.230492   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.658239ms
I0326 14:35:04.274337   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.800072ms
I0326 14:35:04.317801   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.379729ms
I0326 14:35:04.362778   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 44.897882ms
I0326 14:35:04.406845   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.976645ms
I0326 14:35:04.450513   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.613496ms
I0326 14:35:04.494369   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.777934ms
I0326 14:35:04.538399   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.945502ms
I0326 14:35:04.583760   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 45.285171ms
I0326 14:35:04.637430   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 53.629532ms
I0326 14:35:04.681389   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.918124ms
I0326 14:35:04.725401   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.964965ms
I0326 14:35:04.769218   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.734827ms
I0326 14:35:04.812660   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.376494ms
I0326 14:35:04.857974   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 45.246004ms
I0326 14:35:04.901706   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.668478ms
I0326 14:35:04.945372   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.642202ms
I0326 14:35:04.989023   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.619706ms
I0326 14:35:05.033153   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 44.087168ms
I0326 14:35:05.077038   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.791991ms
I0326 14:35:05.124299   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 47.214038ms
I0326 14:35:05.168162   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.795225ms
I0326 14:35:05.211687   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.48304ms
I0326 14:35:05.255553   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.799647ms
I0326 14:35:05.299352   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.72493ms
I0326 14:35:05.342916   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.509589ms
I0326 14:35:05.386952   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.947881ms
I0326 14:35:05.431467   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 44.442041ms
I0326 14:35:05.475834   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 44.304759ms
I0326 14:35:05.519373   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.501574ms
I0326 14:35:05.563584   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 44.162687ms
I0326 14:35:05.607126   12099 serve_hostnames.go:249] Proxy call in namespace serve-hostnames-1631 took 43.478674ms
I0326 14:35:05.607164   12099 serve_hostnames.go:258] serve-hostname-3-0: 12
I0326 14:35:05.607176   12099 serve_hostnames.go:258] serve-hostname-1-0: 10
I0326 14:35:05.607186   12099 serve_hostnames.go:258] serve-hostname-0-0: 18
W0326 14:35:05.607199   12099 serve_hostnames.go:265] No response from pod serve-hostname-2-0 on node kubernetes-node-d0yo.c.kubernetes-satnam.internal at iteration 0
I0326 14:35:05.607211   12099 serve_hostnames.go:269] Iteration 0 took 1.774856469s for 40 queries (22.54 QPS)
I0326 14:35:05.607236   12099 serve_hostnames.go:182] Cleaning up pods
I0326 14:35:05.797893   12099 serve_hostnames.go:130] Cleaning up service serve-hostnames-1631/server-hostnames
```


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/test/soak/serve_hostnames/README.md?pixel)]()
