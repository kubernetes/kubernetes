package benchmarking

import (
	"fmt"
	utilproxy "k8s.io/kubernetes/pkg/proxy/util"
	"testing"
)

func BenchmarkWriter(b *testing.B) {
	nRules := []int{10, 50, 100, 200, 500}

	for _, n := range nRules {
		b.Run(fmt.Sprintf("rule-%d", n), func(b *testing.B) {
			for j := 0; j < b.N; j++ {
				rules := utilproxy.LineBuffer{}
				for i := 0; i < n; i++ {
					//rules.Write(
					//	"-A", "KUBE-SERVICES",
					//	"-m", "comment", "--comment", `"test/default cluster IP"`,
					//	"-m", "tcp", "-p", "tcp",
					//	"-d", "10.96.11.12",
					//	"--dport", "8000",
					//	"-j", "KUBE-SVC-QWERTY")

					rules.Write(
						"-A KUBE-SERVICES",
						"-m comment --comment", `"test/default cluster IP"`,
						"-m tcp -p tcp",
						"-d 10.96.11.12",
						"--dport 8000",
						"-j KUBE-SVC-QWERTY")
				}
			}
		})
	}
}
