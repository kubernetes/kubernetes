# etcd-top
etcd realtime workload analyzer.  Useful for rapid diagnosis of production usage issues and analysis of production request distributions.

usage:
```
  -iface="eth0": interface for sniffing traffic on
  -period=1: seconds between submissions
  -ports="2379,4001": etcd listening ports
  -promiscuous=true: whether to perform promiscuous sniffing or not.
  -topk=10: submit stats for the top <K> sniffed paths
```

result:
```
go run etcd-top.go --period=1 -topk=3
1440035702 sniffed 1074 requests over last 1 seconds

Top 3 most popular http requests:
     Sum     Rate Verb Path
    1305       22 GET /v2/keys/c
    1302        8 GET /v2/keys/S
    1297       10 GET /v2/keys/h
```
