# GCE Load-Balancer Controller (GLBC) Cluster Addon

This cluster addon is composed of:
* A [Google L7 LoadBalancer Controller](https://github.com/kubernetes/contrib/tree/master/ingress/controllers/gce)
* A [404 default backend](https://github.com/kubernetes/contrib/tree/master/404-server) Service + RC

It relies on the [Ingress resource](https://kubernetes.io/docs/user-guide/ingress.md) only available in Kubernetes version 1.1 and beyond.

## Prerequisites

Before you can receive traffic through the GCE L7 Loadbalancer Controller you need:
* A Working Kubernetes 1.1 cluster
* At least 1 Kubernetes [NodePort Service](https://kubernetes.io/docs/user-guide/services.md#type-nodeport) (this is the endpoint for your Ingress)
* Firewall-rules that allow traffic to the NodePort service, as indicated by `kubectl` at Service creation time
* Adequate quota, as mentioned in the next section
* A single instance of the L7 Loadbalancer Controller pod (if you're using the default GCE setup, this should already be running in the `kube-system` namespace)

## Quota

GLBC is not aware of your GCE quota. As of this writing users get 3 [GCE Backend Services](https://cloud.google.com/compute/docs/load-balancing/http/backend-service) by default. If you plan on creating Ingresses for multiple Kubernetes Services, remember that each one requires a backend service, and request quota. Should you fail to do so the controller will poll periodically and grab the first free backend service slot it finds. You can view your quota:

```console
$ gcloud compute project-info describe --project myproject
```
See [GCE documentation](https://cloud.google.com/compute/docs/resource-quotas#checking_your_quota) for how to request more.

## Latency

It takes ~1m to spin up a loadbalancer (this includes acquiring the public ip), and ~5-6m before the GCE api starts healthchecking backends. So as far as latency goes, here's what to expect:

Assume one creates the following simple Ingress:
```yaml
apiVersion: extensions/v1beta1
kind: Ingress
metadata:
  name: test-ingress
spec:
  backend:
    # This will just loopback to the default backend of GLBC
    serviceName: default-http-backend
    servicePort: 80
```

* time, t=0
  ```console
  $ kubectl get ing
  NAME           RULE      BACKEND                   ADDRESS
  test-ingress   -         default-http-backend:80
  $ kubectl describe ing
  No events.
  ```

* time, t=1m
  ```console
  $ kubectl get ing
  NAME           RULE      BACKEND                   ADDRESS
  test-ingress   -         default-http-backend:80   130.211.5.27

  $ kubectl describe ing
  target-proxy:		k8s-tp-default-test-ingress
  url-map:		    k8s-um-default-test-ingress
  backends:		    {"k8s-be-32342":"UNKNOWN"}
  forwarding-rule:	k8s-fw-default-test-ingress
  Events:
    FirstSeen	LastSeen	Count	From				SubobjectPath	Reason	Message
    ─────────	────────	─────	────				─────────────	──────	───────
    46s		46s		1	{loadbalancer-controller }	Success	Created loadbalancer 130.211.5.27
  ```

* time, t=5m
  ```console
  $ kubectl describe ing
  target-proxy:		k8s-tp-default-test-ingress
  url-map:		    k8s-um-default-test-ingress
  backends:		    {"k8s-be-32342":"HEALTHY"}
  forwarding-rule:	k8s-fw-default-test-ingress
  Events:
    FirstSeen	LastSeen	Count	From				SubobjectPath	Reason	Message
    ─────────	────────	─────	────				─────────────	──────	───────
    46s		46s		1	{loadbalancer-controller }	Success	Created loadbalancer 130.211.5.27
  ```

## Disabling GLBC

Since GLBC runs as a cluster addon, you cannot simply delete the RC. The easiest way to disable it is to do as follows:

* IFF you want to tear down existing L7 loadbalancers, hit the /delete-all-and-quit endpoint on the pod:

  ```console
  $ kubectl get pods --namespace=kube-system
  NAME                                               READY     STATUS    RESTARTS   AGE
  l7-lb-controller-7bb21                             1/1       Running   0          1h
  $ kubectl exec l7-lb-controller-7bb21 -c l7-lb-controller curl http://localhost:8081/delete-all-and-quit --namespace=kube-system
  $ kubectl logs l7-lb-controller-7b221 -c l7-lb-controller --follow
  ...
  I1007 00:30:00.322528       1 main.go:160] Handled quit, awaiting pod deletion.
  ```

* Nullify the RC (but don't delete it or the addon controller will "fix" it for you)
  ```console
  $ kubectl scale rc l7-lb-controller --replicas=0 --namespace=kube-system
  ```

## Limitations

* This cluster addon is still in the Beta phase. It behooves you to read through the GLBC documentation mentioned above and make sure there are no surprises.
* The recommended way to tear down a cluster with active Ingresses is to either delete each Ingress, or hit the /delete-all-and-quit endpoint on GLBC as described below, before invoking a cluster teardown script (eg: kube-down.sh). You will have to manually cleanup GCE resources through the [cloud console](https://cloud.google.com/compute/docs/console#access) or [gcloud CLI](https://cloud.google.com/compute/docs/gcloud-compute/) if you simply tear down the cluster with active Ingresses.
* All L7 Loadbalancers created by GLBC have a default backend. If you don't specify one in your Ingress, GLBC will assign the 404 default backend mentioned above.
* All Kubernetes services must serve a 200 page on '/', or whatever custom value you've specified through GLBC's `--health-check-path argument`.
* GLBC is not built for performance. Creating many Ingresses at a time can overwhelm it. It won't fall over, but will take its own time to churn through the Ingress queue. It doesn't understand concepts like fairness or backoff just yet.

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/cluster-loadbalancing/glbc/README.md?pixel)]()
