# sloppy Net parsers

Golang 1.17 released with a necessary breaking change on the net parsers,
rejecting leading zeros in IP addresses.

There is a security CVE associated https://github.com/sickcodes/security/blob/master/advisories/SICK-2021-016.md
because leading zeros can cause parser misalignment between golang and other implementations,
and this be exploited by malicious actors.

Basically the problem is that Golang interprets leading zeros as decimal:

10.008.011.012 = 10.8.11.12

However, other implementations interpret leading zeros as octal, per example, ping or iptables:

```sh
$ iptables -A INPUT -d 10.008.011.012 -m comment --comment test-ip-zeros -j ACCEPT
$ iptables-save | grep test-ip-zero
-A INPUT -d 10.8.9.10/32 -m comment --comment test-ip-zeros -j ACCEPT
```

The Kubernetes project has to keep compatibility with previous stored values, so it has forked
the previous golang parsers as part of "k8s.io/utils/net": ParseIPSloppy() and ParseCIDRSloppy()

https://github.com/kubernetes/utils/pull/207


However, refactoring the code to replace the standard lib network parsers is complex, even more complicated
on networking projects, where these functions are heavily used.

This tool converts the standard library functions to the forked ones, it can work on files or on paths, replacing
every occurrence of net.ParseIP or net.ParseCIDR by its previous versions (1.16-), and fixing the imports
accordenly.

It can rewrite the file directly or just output the difference without doing any modification

```sh
$ sloppy-netparser -diff cmd/kubeadm/app/constants/constants.go

real    0m0,007s
user    0m0,006s
sys     0m0,001s
[aojea@juanan kubernetes]$ ^C
[aojea@juanan kubernetes]$ git restore cmd/kubeadm/app/constants/constants.go
[aojea@juanan kubernetes]$ ../../github.com/aojea/sloppy-netparser/sloppy-netparser -diff cmd/kubeadm/app/constants/constants.go
cmd/kubeadm/app/constants/constants.go: fixed sloppy-netparsers
diff cmd/kubeadm/app/constants/constants.go fixed/cmd/kubeadm/app/constants/constants.go
--- /tmp/go-fix586224454        2021-08-19 19:38:16.170615958 +0200
+++ /tmp/go-fix437319661        2021-08-19 19:38:16.170615958 +0200
@@ -34,7 +34,7 @@
        apimachineryversion "k8s.io/apimachinery/pkg/version"
        bootstrapapi "k8s.io/cluster-bootstrap/token/api"
        componentversion "k8s.io/component-base/version"
-       utilnet "k8s.io/utils/net"
+       netutils "k8s.io/utils/net"
 )
 
 const (
@@ -635,7 +635,7 @@
        }
 
        // Selects the 10th IP in service subnet CIDR range as dnsIP
-       dnsIP, err := utilnet.GetIndexedIP(svcSubnetCIDR, 10)
+       dnsIP, err := netutils.GetIndexedIP(svcSubnetCIDR, 10)
        if err != nil {
                return nil, errors.Wrap(err, "unable to get internal Kubernetes Service IP from the given service CIDR")
        }
@@ -649,7 +649,7 @@
                // The default service address family for the cluster is the address family of the first
                // service cluster IP range configured via the `--service-cluster-ip-range` flag
                // of the kube-controller-manager and kube-apiserver.
-               svcSubnets, err := utilnet.ParseCIDRs(strings.Split(svcSubnetList, ","))
+               svcSubnets, err := netutils.ParseCIDRs(strings.Split(svcSubnetList, ","))
                if err != nil {
                        return nil, errors.Wrapf(err, "unable to parse ServiceSubnet %v", svcSubnetList)
                }
@@ -659,7 +659,7 @@
                return svcSubnets[0], nil
        }
        // internal IP address for the API server
-       _, svcSubnet, err := net.ParseCIDR(svcSubnetList)
+       _, svcSubnet, err := netutils.ParseCIDRSloppy(svcSubnetList)
        if err != nil {
                return nil, errors.Wrapf(err, "unable to parse ServiceSubnet %v", svcSubnetList)
        }
@@ -672,7 +672,7 @@
        if err != nil {
                return nil, errors.Wrap(err, "unable to get internal Kubernetes Service IP from the given service CIDR")
        }
-       internalAPIServerVirtualIP, err := utilnet.GetIndexedIP(svcSubnet, 1)
+       internalAPIServerVirtualIP, err := netutils.GetIndexedIP(svcSubnet, 1)
        if err != nil {
                return nil, errors.Wrapf(err, "unable to get the first IP address from the given CIDR: %s", svcSubnet.String())
        }
```