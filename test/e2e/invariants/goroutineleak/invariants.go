/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// please speak to SIG-Testing leads before adding anything to this package
// see: https://git.k8s.io/enhancements/keps/sig-testing/5468-invariant-testing

// Package goroutineleak collects the goroutineleak profile added in Go 1.27
// from cluster components and reports permanently leaked goroutines.
//
// The runtime reports a goroutine only when it is blocked on a synchronization
// primitive which is unreachable from any live goroutine, so nothing can ever
// unblock it. Goroutines waiting on network I/O, timers, tickers or channels
// which are still reachable are not reported, so the usual worker and informer
// patterns do not produce findings.
//
// See https://go.dev/blog/goroutine-leak-profiles.
package goroutineleak

import (
	"context"
	"fmt"
	"net"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
)

const (
	// profilePath is the endpoint served by net/http/pprof for the goroutine
	// leak profile added in Go 1.27. Components register it implicitly: both
	// routes.Profiling{}.Install and the API server's profiling routes install
	// pprof.Index on the whole /debug/pprof/ prefix, which dispatches unknown
	// profile names to pprof.Lookup.
	profilePath = "debug/pprof/goroutineleak"

	// scrapeTimeout bounds a single component scrape so that one unresponsive
	// component cannot stall the suite.
	scrapeTimeout = 30 * time.Second

	// kubeSchedulerPort and kubeControllerManagerPort are the default status
	// server ports, the same values used by the e2e metrics grabber.
	kubeSchedulerPort         = 10259
	kubeControllerManagerPort = 10257
)

var (
	// Control plane pods are matched by name, the same way the e2e metrics
	// grabber discovers them, because kubeadm appends the node name to static
	// pod names. There is no API for identifying control plane components,
	// see https://github.com/kubernetes/enhancements/issues/5708.
	regKubeScheduler         = regexp.MustCompile("kube-scheduler-.*")
	regKubeControllerManager = regexp.MustCompile("kube-controller-manager-.*")
)

// Owners identifies who is responsible for a finding. KEP-5468 requires that
// every invariant has documented owners which are surfaced with the result.
type Owners struct {
	// SIG associated with the invariant, without the "sig-" prefix.
	SIG string
	// Owners are the GitHub handles to assign bugs to.
	Owners []string
}

func (o Owners) String() string {
	return fmt.Sprintf("SIG: %s, Owners: %s", o.SIG, strings.Join(o.Owners, ", "))
}

// defaultOwners is used until per-component ownership is agreed with the
// owning SIGs.
var defaultOwners = Owners{
	SIG:    "testing",
	Owners: []string{"pohly"},
}

// PodDialer establishes a connection to a port inside a pod. The e2e suite
// provides an implementation backed by the pod portforward subresource, which
// is how the metrics grabber reaches components that listen on localhost.
type PodDialer interface {
	DialPod(ctx context.Context, namespace, podName string, port int) (net.Conn, error)
}

// Leak is one group of goroutines sharing an identical stack.
type Leak struct {
	// Count is how many goroutines share this stack.
	Count int
	// Function is the innermost named function of the stack, if it could be
	// determined.
	Function string
	// Location is "file:line" for Function, if it could be determined.
	Location string
}

// Result is the outcome of scraping a single component.
type Result struct {
	// Component identifies what was scraped, for example "kube-apiserver" or
	// "kubelet/node-1".
	Component string
	// Total is the number of leaked goroutines reported by the runtime.
	Total int
	// Leaks are the distinct stacks, most frequent first.
	Leaks []Leak
	// Err is non-nil when the component could not be scraped. This is not a
	// finding: profiling may be disabled, or the component may not be
	// reachable in this cluster topology.
	Err error
}

var (
	totalRE = regexp.MustCompile(`goroutineleak profile: total (\d+)`)
	// A stack header line, for example "3 @ 0x48c1aa 0x419c2e".
	headerRE = regexp.MustCompile(`^(\d+) @`)
	// A frame line, for example "#\t0x67d744\tmain.leak+0x24\t/path/main.go:13".
	frameRE = regexp.MustCompile(`^#\s+0x[0-9a-f]+\s+(\S+)\s+(\S+)$`)
)

// Parse turns the debug=1 text form of the goroutineleak profile into a
// Result. The text form is used rather than the binary profile because it
// carries a total, a per-stack count and a file:line, which is what makes a
// finding actionable.
func Parse(component string, body []byte) (Result, error) {
	res := Result{Component: component}

	m := totalRE.FindSubmatch(body)
	if m == nil {
		return res, fmt.Errorf("unrecognized goroutineleak profile for %s: missing total line", component)
	}
	total, err := strconv.Atoi(string(m[1]))
	if err != nil {
		return res, fmt.Errorf("parsing total for %s: %w", component, err)
	}
	res.Total = total

	var current *Leak
	for line := range strings.SplitSeq(string(body), "\n") {
		if h := headerRE.FindStringSubmatch(line); h != nil {
			n, err := strconv.Atoi(h[1])
			if err != nil {
				continue
			}
			res.Leaks = append(res.Leaks, Leak{Count: n})
			current = &res.Leaks[len(res.Leaks)-1]
			continue
		}
		// Record the first frame of each stack: the innermost named function,
		// which is where the goroutine is blocked.
		if current != nil && current.Function == "" {
			if f := frameRE.FindStringSubmatch(line); f != nil {
				current.Function = f[1]
				current.Location = f[2]
			}
		}
	}

	sort.SliceStable(res.Leaks, func(i, j int) bool { return res.Leaks[i].Count > res.Leaks[j].Count })
	return res, nil
}

func result(component string, body []byte, err error) Result {
	if err != nil {
		return Result{Component: component, Err: fmt.Errorf("scraping %s: %w", profilePath, err)}
	}
	res, err := Parse(component, body)
	if err != nil {
		return Result{Component: component, Err: err}
	}
	return res
}

// CheckAPIServer collects the leak profile from the API server, the same way
// the metrics invariant scrapes its metrics endpoint.
func CheckAPIServer(ctx context.Context, client clientset.Interface) Result {
	ctx, cancel := context.WithTimeout(ctx, scrapeTimeout)
	defer cancel()

	body, err := client.Discovery().RESTClient().Get().
		AbsPath(profilePath).
		Param("debug", "1").
		DoRaw(ctx)
	return result("kube-apiserver", body, err)
}

// CheckKubelets collects the leak profile from every node matching nodes, via
// the node proxy. This is the mechanism the e2e framework already uses to
// fetch a heap profile from the kubelet.
func CheckKubelets(ctx context.Context, client clientset.Interface, nodes *regexp.Regexp) []Result {
	nodeList, err := client.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		return []Result{{Component: "kubelet", Err: fmt.Errorf("listing nodes: %w", err)}}
	}

	var results []Result
	for _, node := range nodeList.Items {
		if nodes != nil && !nodes.MatchString(node.Name) {
			continue
		}
		// Prefer the port the node advertises over the default, as the
		// metrics grabber does.
		port := int(node.Status.DaemonEndpoints.KubeletEndpoint.Port)
		if port <= 0 || port > 65535 {
			results = append(results, Result{
				Component: "kubelet/" + node.Name,
				Err:       fmt.Errorf("invalid kubelet port %d", port),
			})
			continue
		}
		results = append(results, checkKubelet(ctx, client, node.Name, port))
	}
	return results
}

func checkKubelet(ctx context.Context, client clientset.Interface, nodeName string, port int) Result {
	ctx, cancel := context.WithTimeout(ctx, scrapeTimeout)
	defer cancel()

	body, err := client.CoreV1().RESTClient().Get().
		Resource("nodes").
		SubResource("proxy").
		Name(fmt.Sprintf("%v:%v", nodeName, port)).
		Suffix(profilePath).
		Param("debug", "1").
		DoRaw(ctx)
	return result("kubelet/"+nodeName, body, err)
}

// CheckControlPlanePods collects the leak profile from kube-controller-manager
// and kube-scheduler.
//
// Those components bind to localhost and authorize with a delegating
// authorizer, so the API server's pod proxy can neither reach them nor present
// an accepted identity. Connecting through the pod portforward subresource
// while keeping the suite's own credentials solves both, which is how the
// metrics grabber reaches their metrics endpoints.
func CheckControlPlanePods(ctx context.Context, client clientset.Interface, config *restclient.Config, dialer PodDialer) []Result {
	pods, err := client.CoreV1().Pods(metav1.NamespaceSystem).List(ctx, metav1.ListOptions{})
	if err != nil {
		return []Result{{Component: "control plane pods", Err: fmt.Errorf("listing pods in %s: %w", metav1.NamespaceSystem, err)}}
	}

	var results []Result
	for _, pod := range pods.Items {
		var port int
		switch {
		case regKubeControllerManager.MatchString(pod.Name):
			port = kubeControllerManagerPort
		case regKubeScheduler.MatchString(pod.Name):
			port = kubeSchedulerPort
		default:
			continue
		}
		results = append(results, checkControlPlanePod(ctx, config, dialer, pod.Namespace, pod.Name, port))
	}
	return results
}

func checkControlPlanePod(ctx context.Context, config *restclient.Config, dialer PodDialer, namespace, name string, port int) Result {
	component := namespace + "/" + name

	ctx, cancel := context.WithTimeout(ctx, scrapeTimeout)
	defer cancel()

	// Only the dial is replaced, so the request still carries the suite's own
	// credentials and the delegating authorizer accepts it. The component
	// serves a certificate which is not signed by the cluster CA, so
	// verification is disabled, as the metrics grabber does.
	profileConfig := restclient.CopyConfig(config)
	profileConfig.Dial = func(ctx context.Context, network, address string) (net.Conn, error) {
		return dialer.DialPod(ctx, namespace, name, port)
	}
	profileConfig.Host = fmt.Sprintf("%s.%s:%d", namespace, name, port)
	profileConfig.ServerName = "localhost"
	profileConfig.Insecure = true
	profileConfig.CAFile = ""
	profileConfig.CAData = nil

	profileClient, err := clientset.NewForConfig(profileConfig)
	if err != nil {
		return Result{Component: component, Err: fmt.Errorf("building client: %w", err)}
	}

	body, err := profileClient.RESTClient().Get().
		AbsPath(profilePath).
		Param("debug", "1").
		DoRaw(ctx)
	return result(component, body, err)
}

// Report renders results as human readable text. It always lists what was
// checked, including components which reported no leaks, so that a check which
// examined nothing is distinguishable from one which passed.
func Report(results []Result) string {
	var b strings.Builder
	var checked, skipped []string

	for _, r := range results {
		switch {
		case r.Err != nil:
			skipped = append(skipped, fmt.Sprintf("%s (not checked: %v)", r.Component, r.Err))
		case r.Total == 0:
			checked = append(checked, fmt.Sprintf("%s (ok)", r.Component))
		default:
			checked = append(checked, fmt.Sprintf("%s (%d leaked)", r.Component, r.Total))
		}
	}

	fmt.Fprintf(&b, "Checked: %s\n", strings.Join(checked, ", "))
	if len(skipped) > 0 {
		fmt.Fprintf(&b, "Skipped: %s\n", strings.Join(skipped, ", "))
	}

	for _, r := range results {
		if r.Err != nil || r.Total == 0 {
			continue
		}
		fmt.Fprintf(&b, "\n%d leaked goroutine(s) in %s (%s)\n", r.Total, r.Component, defaultOwners)
		for _, l := range r.Leaks {
			if l.Function == "" {
				fmt.Fprintf(&b, "  %d x <unknown stack>\n", l.Count)
				continue
			}
			fmt.Fprintf(&b, "  %d x %s\n      %s\n", l.Count, l.Function, l.Location)
		}
	}
	return b.String()
}

// Failure returns a message describing the findings, or an empty string if
// there are none. Components which could not be scraped are not findings.
func Failure(results []Result) string {
	var total int
	for _, r := range results {
		if r.Err == nil {
			total += r.Total
		}
	}
	if total == 0 {
		return ""
	}
	return fmt.Sprintf(`goroutine leak invariant failed: %d leaked goroutine(s)

%s
If this failed on a pull request, please check if the PR changes may be related to the failure.
If not, you can also search for an existing GitHub issue before filing a new issue.

If this failed in a periodic CI job, please file a bug and /assign the owners`,
		total, Report(results))
}
