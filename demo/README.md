### Prep

make quick-release
local-controlplane-up

### Sample Controller Demo

cd staging/src/k8s.io/sample-controller
go build -o sample-controller .

k apply -f staging/src/k8s.io/sample-controller/artifacts/examples/crd.yaml

k get -n kube-system leases sample-controller -oyaml --watch

./sample-controller --kubeconfig=/var/run/kubernetes/admin.kubeconfig --binary-version=1.29 --compatibility-version=1.29 --identity=sample-controller-a
# wait for a to claim lease

./sample-controller --kubeconfig=/var/run/kubernetes/admin.kubeconfig --binary-version=1.28 --compatibility-version=1.28 --identity=sample-controller-b
# observe how b takes over, since it has lower version

### Old: KCM controller demo (don't use until election controller moved to apiserver)

k apply -f demo
k get -n kube-system leases
k get -n kube-system lease amazing-controller -oyaml
k delete -n kube-system lease amazing-controller

k get -n kube-system lease kube-controller-manager -oyaml
# note that the controller is renewing..


### Rules

- Controllers (in leaderelection.go)
- If isLeader AND lease is not expired AND not end-of-term: renew
- never acquire directly

### TODO

- Keep identity leases alive, don't attempt to elect if identity lease is stale
- Don't pick a leader from equivalents, instead, favor keeping the current leader
- Funnel all reconciliation to main lease. If a identity lease is reconciled, enqueue the main lease for reconciliation..
- Move election controller to apiserver
- [x] Create a variant of leaselock to do coordinated elections
- Create a variant of leaderelection that can be used for coorindated elections
- How to prevent election controller churn where there are two views of candidates

rm ~/demo.cast
asciinema rec ~/demo.cast -i 2 -c "PS1='$ ' sh"

# Coordinated leader election prefers Kubernetes controller manager instances running at older versions
# To demonstrate this, we will first start a controller manager running at version 1.30:

# Next, we will start a controller running at 1.29:

# Note how the controller claimed the lease, even though the 1.30 controller manager kept running

# Now if we restart the 1.30 controller manager, it won't take the lease

# But if we stop the 1.29 controller manager, the 1.30 manager is the only one still running, so it
# claims the lease