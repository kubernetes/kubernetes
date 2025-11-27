Feature 1: kubectl get pods --stale (one-liner that people actually want)
What it does
Shows only pods that haven’t received traffic or been accessed in X minutes/hours (useful in large clusters to find forgotten pods).
Scope – under 300 lines total

Add one new flag --stale=duration to cmd/kubectl/kubectl get/get.go
Reuse the existing LastTransitionTime from conditions OR fall back to pod’s status.startTime
New table column “STALE” that prints 2h, 15m, <1m
Unit tests in staging/src/k8s.io/kubectl/pkg/cmd/get
One e2e test in test/cmd/get.sh
Update the kubectl reference docs (one paragraph)

Total time: 4–8 hours.
This has been requested literally hundreds of times and will never get rejected for being “too small” in the challenge because it’s useful and touches the real CLI.