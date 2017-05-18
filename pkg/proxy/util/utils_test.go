package util

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/proxy"
)

func makeNSN(namespace, name string) types.NamespacedName {
	return types.NamespacedName{Namespace: namespace, Name: name}
}

func makeServicePortName(ns, name, port string) proxy.ServicePortName {
	return proxy.ServicePortName{
		NamespacedName: makeNSN(ns, name),
		Port:           port,
	}
}

func Test_getLocalIPs(t *testing.T) {
	testCases := []struct {
		endpointsMap map[proxy.ServicePortName][]*EndpointsInfo
		expected     map[types.NamespacedName]sets.String
	}{{
		// Case[0]: nothing
		endpointsMap: map[proxy.ServicePortName][]*EndpointsInfo{},
		expected:     map[types.NamespacedName]sets.String{},
	}, {
		// Case[1]: unnamed port
		endpointsMap: map[proxy.ServicePortName][]*EndpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{"1.1.1.1:11", false},
			},
		},
		expected: map[types.NamespacedName]sets.String{},
	}, {
		// Case[2]: unnamed port local
		endpointsMap: map[proxy.ServicePortName][]*EndpointsInfo{
			makeServicePortName("ns1", "ep1", ""): {
				{"1.1.1.1:11", true},
			},
		},
		expected: map[types.NamespacedName]sets.String{
			{Namespace: "ns1", Name: "ep1"}: sets.NewString("1.1.1.1"),
		},
	}, {
		// Case[3]: named local and non-local ports for the same IP.
		endpointsMap: map[proxy.ServicePortName][]*EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{"1.1.1.1:11", false},
				{"1.1.1.2:11", true},
			},
			makeServicePortName("ns1", "ep1", "p12"): {
				{"1.1.1.1:12", false},
				{"1.1.1.2:12", true},
			},
		},
		expected: map[types.NamespacedName]sets.String{
			{Namespace: "ns1", Name: "ep1"}: sets.NewString("1.1.1.2"),
		},
	}, {
		// Case[4]: named local and non-local ports for different IPs.
		endpointsMap: map[proxy.ServicePortName][]*EndpointsInfo{
			makeServicePortName("ns1", "ep1", "p11"): {
				{"1.1.1.1:11", false},
			},
			makeServicePortName("ns2", "ep2", "p22"): {
				{"2.2.2.2:22", true},
				{"2.2.2.22:22", true},
			},
			makeServicePortName("ns2", "ep2", "p23"): {
				{"2.2.2.3:23", true},
			},
			makeServicePortName("ns4", "ep4", "p44"): {
				{"4.4.4.4:44", true},
				{"4.4.4.5:44", false},
			},
			makeServicePortName("ns4", "ep4", "p45"): {
				{"4.4.4.6:45", true},
			},
		},
		expected: map[types.NamespacedName]sets.String{
			{Namespace: "ns2", Name: "ep2"}: sets.NewString("2.2.2.2", "2.2.2.22", "2.2.2.3"),
			{Namespace: "ns4", Name: "ep4"}: sets.NewString("4.4.4.4", "4.4.4.6"),
		},
	}}

	for tci, tc := range testCases {
		// outputs
		localIPs := getLocalIPs(tc.endpointsMap)

		if !reflect.DeepEqual(localIPs, tc.expected) {
			t.Errorf("[%d] expected %#v, got %#v", tci, tc.expected, localIPs)
		}
	}
}
