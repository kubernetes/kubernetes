# Reproduction of k8s issue #74839

Network services with heavy load will cause "connection reset" from time to
time. Especially those with big payloads. When packets with sequence number
out-of-window arrived k8s node, conntrack marked them as INVALID. kube-proxy
will ignore them, without rewriting DNAT. The packet goes back the the original
pod, who doesn't recognize the packet because of the wrong source ip, end up
RSTing the connection.

## Reference

https://github.com/kubernetes/kubernetes/issues/74839


