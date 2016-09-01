<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

### explorer

Explorer is a little container for examining the runtime environment kubernetes produces for your pods.

The intended use is to substitute gcr.io/google_containers/explorer for your intended container, and then visit it via the proxy.

Currently, you can look at:
 * The environment variables to make sure kubernetes is doing what you expect.
 * The filesystem to make sure the mounted volumes and files are also what you expect.
 * Perform DNS lookups, to see how DNS works.

`pod.yaml` is supplied as an example. You can control the port it serves on with the -port flag.

Example from command line (the DNS lookup looks better from a web browser):

```console
$ kubectl create -f examples/explorer/pod.yaml
$ kubectl proxy &
Starting to serve on localhost:8001

$ curl localhost:8001/api/v1/proxy/namespaces/default/pods/explorer:8080/vars/
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
HOSTNAME=explorer
KIBANA_LOGGING_PORT_5601_TCP_PORT=5601
KUBERNETES_SERVICE_HOST=10.0.0.2
MONITORING_GRAFANA_PORT_80_TCP_PROTO=tcp
MONITORING_INFLUXDB_UI_PORT_80_TCP_PROTO=tcp
KIBANA_LOGGING_SERVICE_PORT=5601
MONITORING_HEAPSTER_PORT_80_TCP_PORT=80
MONITORING_INFLUXDB_UI_PORT_80_TCP_PORT=80
KIBANA_LOGGING_SERVICE_HOST=10.0.204.206
KIBANA_LOGGING_PORT_5601_TCP=tcp://10.0.204.206:5601
KUBERNETES_PORT=tcp://10.0.0.2:443
MONITORING_INFLUXDB_PORT=tcp://10.0.2.30:80
MONITORING_INFLUXDB_PORT_80_TCP_PROTO=tcp
MONITORING_INFLUXDB_UI_PORT=tcp://10.0.36.78:80
KUBE_DNS_PORT_53_UDP=udp://10.0.0.10:53
MONITORING_INFLUXDB_SERVICE_HOST=10.0.2.30
ELASTICSEARCH_LOGGING_PORT=tcp://10.0.48.200:9200
ELASTICSEARCH_LOGGING_PORT_9200_TCP_PORT=9200
KUBERNETES_PORT_443_TCP=tcp://10.0.0.2:443
ELASTICSEARCH_LOGGING_PORT_9200_TCP_PROTO=tcp
KIBANA_LOGGING_PORT_5601_TCP_ADDR=10.0.204.206
KUBE_DNS_PORT_53_UDP_ADDR=10.0.0.10
MONITORING_HEAPSTER_PORT_80_TCP_PROTO=tcp
MONITORING_INFLUXDB_PORT_80_TCP_ADDR=10.0.2.30
KIBANA_LOGGING_PORT=tcp://10.0.204.206:5601
MONITORING_GRAFANA_SERVICE_PORT=80
MONITORING_HEAPSTER_SERVICE_PORT=80
MONITORING_HEAPSTER_PORT_80_TCP=tcp://10.0.150.238:80
ELASTICSEARCH_LOGGING_PORT_9200_TCP=tcp://10.0.48.200:9200
ELASTICSEARCH_LOGGING_PORT_9200_TCP_ADDR=10.0.48.200
MONITORING_GRAFANA_PORT_80_TCP_PORT=80
MONITORING_HEAPSTER_PORT=tcp://10.0.150.238:80
MONITORING_INFLUXDB_PORT_80_TCP=tcp://10.0.2.30:80
KUBE_DNS_SERVICE_PORT=53
KUBE_DNS_PORT_53_UDP_PORT=53
MONITORING_GRAFANA_PORT_80_TCP_ADDR=10.0.100.174
MONITORING_INFLUXDB_UI_SERVICE_HOST=10.0.36.78
KIBANA_LOGGING_PORT_5601_TCP_PROTO=tcp
MONITORING_GRAFANA_PORT=tcp://10.0.100.174:80
MONITORING_INFLUXDB_UI_PORT_80_TCP_ADDR=10.0.36.78
KUBE_DNS_SERVICE_HOST=10.0.0.10
KUBERNETES_PORT_443_TCP_PORT=443
MONITORING_HEAPSTER_PORT_80_TCP_ADDR=10.0.150.238
MONITORING_INFLUXDB_UI_SERVICE_PORT=80
KUBE_DNS_PORT=udp://10.0.0.10:53
ELASTICSEARCH_LOGGING_SERVICE_HOST=10.0.48.200
KUBERNETES_SERVICE_PORT=443
MONITORING_HEAPSTER_SERVICE_HOST=10.0.150.238
MONITORING_INFLUXDB_SERVICE_PORT=80
MONITORING_INFLUXDB_PORT_80_TCP_PORT=80
KUBE_DNS_PORT_53_UDP_PROTO=udp
MONITORING_GRAFANA_PORT_80_TCP=tcp://10.0.100.174:80
ELASTICSEARCH_LOGGING_SERVICE_PORT=9200
MONITORING_GRAFANA_SERVICE_HOST=10.0.100.174
MONITORING_INFLUXDB_UI_PORT_80_TCP=tcp://10.0.36.78:80
KUBERNETES_PORT_443_TCP_PROTO=tcp
KUBERNETES_PORT_443_TCP_ADDR=10.0.0.2
HOME=/

$ curl localhost:8001/api/v1/proxy/namespaces/default/pods/explorer:8080/fs/
mount/
var/
.dockerenv
etc/
dev/
proc/
.dockerinit
sys/
README.md
explorer

$ curl localhost:8001/api/v1/proxy/namespaces/default/pods/explorer:8080/dns?q=elasticsearch-logging
<html><head></head><body>
<form action="/api/v1/proxy/namespaces/default/pods/explorer:8080/dns">
<input name="q" type="text" value="elasticsearch-logging"/>
<button type="submit">Lookup</button>
</form>
<br/><br/><pre>LookupNS(elasticsearch-logging):
Result: ([]*net.NS)<nil>
Error: &lt;*&gt;lookup elasticsearch-logging: no such host

LookupTXT(elasticsearch-logging):
Result: ([]string)<nil>
Error: &lt;*&gt;lookup elasticsearch-logging: no such host

LookupSRV(&#34;&#34;, &#34;&#34;, elasticsearch-logging):
cname: elasticsearch-logging.default.svc.cluster.local.
Result: ([]*net.SRV)[&lt;*&gt;{Target:(string)elasticsearch-logging.default.svc.cluster.local. Port:(uint16)9200 Priority:(uint16)10 Weight:(uint16)100}]
Error: <nil>

LookupHost(elasticsearch-logging):
Result: ([]string)[10.0.60.245]
Error: <nil>

LookupIP(elasticsearch-logging):
Result: ([]net.IP)[10.0.60.245]
Error: <nil>

LookupMX(elasticsearch-logging):
Result: ([]*net.MX)<nil>
Error: &lt;*&gt;lookup elasticsearch-logging: no such host

</nil></nil></nil></nil></nil></nil></pre>

</body></html>
```




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/explorer/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
