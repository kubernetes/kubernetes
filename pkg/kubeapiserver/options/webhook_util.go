package options

import (
	"context"
	"net"
	"strconv"
	"strings"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/klog/v2"
)

// newWebhookDialer return a DialFunc
func newWebhookDialer(serviceResolver webhook.ServiceResolver, baseDialer utilnet.DialFunc) utilnet.DialFunc {
	var d net.Dialer
	delegateDialer := d.DialContext
	if baseDialer != nil {
		delegateDialer = baseDialer
	}
	return func(ctx context.Context, network, addr string) (net.Conn, error) {
		if serviceResolver == nil {
			return delegateDialer(ctx, network, addr)
		}
		var (
			host, port string
			err        error
		)
		host, port, err = net.SplitHostPort(addr)
		if err != nil {
			klog.V(6).Infof("splitting hostport %s error: %v", addr, err)
			return delegateDialer(ctx, network, addr)
		}

		portInt, err := strconv.Atoi(port)
		if err != nil {
			klog.V(6).Infof("port %s convert string to int failed, err: %v", port, err)
			return delegateDialer(ctx, network, addr)
		}
		segs := strings.Split(host, ".")
		if len(segs) < 3 {
			klog.V(6).Infof("dialer address %s is not a kubernetes service reference", host)
			return delegateDialer(ctx, network, addr)
		}
		if segs[2] != "svc" {
			return delegateDialer(ctx, network, addr)
		}
		serviceNamespace := segs[1]
		serviceName := segs[0]

		u, err := serviceResolver.ResolveEndpoint(serviceNamespace, serviceName, int32(portInt))
		if err != nil {
			klog.V(5).Infof("resolving endpoint from namespace: %s, name: %s, original host: %s failed, err: %v",
				serviceNamespace, serviceName, host, err)
			return delegateDialer(ctx, network, addr)
		}

		klog.V(6).Infof("resolved addr %s to hostport: %s:%d",
			addr, u.Host, portInt)
		return delegateDialer(ctx, network, u.Host)
	}
}
