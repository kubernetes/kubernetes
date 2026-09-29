# Kubernetes Scheduler (`pkg/scheduler`)

The Kubernetes Scheduler (`kube-scheduler`) is the core control plane component responsible for assigning unscheduled Pods to suitable Nodes in a cluster.

For a comprehensive guide on the architecture, pod lifecycle, framework extension points, backend caching and queuing subsystems, and developer/agent invariants, see [AGENTS.md](AGENTS.md).

For specifications, summaries, and architecture documents on preemption, gang scheduling, and workload enhancement proposals (KEPs), see the [KEP Documentation Index](keps/README.md).
