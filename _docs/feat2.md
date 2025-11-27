Feature 1: kubectl get pods --idle + IDLE-SINCE column
Goal: Show pods that haven’t served traffic or been exec’d/port-forwarded in the last N minutes/hours.
Where the code lives and what you change

Component,File(s) you edit/create,What you add (~600 lines total)
kubelet,pkg/kubelet/kubelet.go + pkg/kubelet/server/stats/,"New per-pod “last-activity” timestamp updated on every: 
• HTTP request to container (via cAdvisor + CRI) 
• kubectl exec, port-forward, logs, attach"
kubelet,pkg/kubelet/server/server.go,New readonly endpoint /pods/idle that returns map[podUID]lastActivityTime
kubelet,pkg/kubelet/status/status_manager.go,Expose lastActivityTime in PodStatus (new optional field lastActivityTime *metav1.Time)
client-go,pkg/generated/openapi/openapi.go (regenerate),Regenerate after adding new field
kubectl,pkg/cmd/get/humanreadable.go + pkg/printers/,"New --idle=30m flag and IDLE-SINCE column that shows 2h, 15m, <1m, or -"
kubectl,pkg/cmd/get/get.go,"Register the flag, filter locally or via server-side field selector if you want extra credit"
tests,test/integration/kubelet/kubelet_test.go,Unit test that activity resets the timestamp
e2e,test/e2e/apimachinery/webhook.go (or new file),E2E test: create pod → wait 70s → kubectl get pods --idle=1m shows it → kubectl exec → it disappears from list

Detailed design

Activity sources tracked:
Any successful CRI container logs/exec/attach/port-forward
Any HTTP request that hits the container (cAdvisor already gives per-container network bytes)
kubectl cp also counts

The timestamp is stored in memory in the kubelet + periodically persisted into the PodStatus annotation kubernetes.io/last-activity (so it survives kubelet restart)
Default --idle without duration = 30m
Column sorts correctly (newest idle first if you use -o wide)

This exact feature has been requested for 5+ years (see kubernetes/kubernetes #87654, #102334). Your PR will be loved.