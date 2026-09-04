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

// Package goroutineleak collects the Go 1.27 "goroutineleak" profile from
// cluster components and reports permanently leaked goroutines.
//
// A goroutine is reported by the runtime only when it is blocked on a
// synchronization primitive that is unreachable from any live goroutine,
// which means nothing can ever unblock it. Goroutines waiting on network
// I/O, timers, tickers or channels which are still reachable are not
// reported, so the common Kubernetes worker and informer patterns do not
// produce findings.
//
// See https://go.dev/blog/goroutine-leak-profiles.
package goroutineleak

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
)

const (
	// profilePath is the endpoint served by net/http/pprof for the
	// goroutine leak profile added in Go 1.27.
	profilePath = "debug/pprof/goroutineleak"

	// scrapeTimeout bounds a single component scrape. One unresponsive
	// component must not stall the whole suite.
	scrapeTimeout = 30 * time.Second

	// kubeletPort is the default kubelet server port. It is duplicated here
	// rather than imported from the e2e framework so that this package stays
	// usable outside the e2e suite, as the metrics invariant package is.
	kubeletPort = 10250
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
	// Component identifies what was scraped, e.g. "kube-apiserver" or
	// "kubelet/node-1".
	Component string
	// Total is the number of leaked goroutines reported by the runtime.
	Total int
	// Leaks are the distinct stacks, most frequent first.
	Leaks []Leak
	// Err is non-nil when the component could not be scraped. This is not
	// treated as a finding: a component may predate Go 1.27, or may not be
	// reachable in this cluster topology.
	Err error
}

// Checked reports whether the component actually produced a profile.
func (r Result) Checked() bool { return r.Err == nil }

var (
	totalRE = regexp.MustCompile(`goroutineleak profile: total (\d+)`)
	// A stack header line, e.g. "3 @ 0x48c1aa 0x419c2e".
	headerRE = regexp.MustCompile(`^(\d+) @`)
	// A frame line, e.g. "#\t0x67d744\tmain.leakForever+0x24\t/path/main.go:13".
	frameRE = regexp.MustCompile(`^#\s+0x[0-9a-f]+\s+(\S+)\s+(\S+)$`)
)

// Parse turns the debug=1 text form of the goroutineleak profile into a
// Result. The text form is used rather than the binary profile because it is
// self-describing: it carries a total, a per-stack count and a file:line,
// which is what makes a failure message actionable.
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
	for _, line := range strings.Split(string(body), "\n") {
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

// CheckAPIServer collects the leak profile from the API server.
func CheckAPIServer(ctx context.Context, client clientset.Interface) Result {
	ctx, cancel := context.WithTimeout(ctx, scrapeTimeout)
	defer cancel()

	body, err := client.Discovery().RESTClient().Get().
		AbsPath(profilePath).
		Param("debug", "1").
		DoRaw(ctx)
	if err != nil {
		return Result{Component: "kube-apiserver", Err: fmt.Errorf("scraping %s: %w", profilePath, err)}
	}
	res, err := Parse("kube-apiserver", body)
	if err != nil {
		return Result{Component: "kube-apiserver", Err: err}
	}
	return res
}

// CheckKubelets collects the leak profile from every node whose name matches
// nodes, via the node proxy. This mirrors how the e2e framework already
// fetches a heap profile from the kubelet.
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
		results = append(results, checkKubelet(ctx, client, node.Name))
	}
	return results
}

func checkKubelet(ctx context.Context, client clientset.Interface, nodeName string) Result {
	component := "kubelet/" + nodeName

	ctx, cancel := context.WithTimeout(ctx, scrapeTimeout)
	defer cancel()

	body, err := client.CoreV1().RESTClient().Get().
		Resource("nodes").
		SubResource("proxy").
		Name(fmt.Sprintf("%v:%v", nodeName, kubeletPort)).
		Suffix(profilePath).
		Param("debug", "1").
		DoRaw(ctx)
	if err != nil {
		return Result{Component: component, Err: fmt.Errorf("scraping %s: %w", profilePath, err)}
	}
	res, err := Parse(component, body)
	if err != nil {
		return Result{Component: component, Err: err}
	}
	return res
}

// CheckPods collects the leak profile from control plane pods in namespaces
// matching namespaces, via the pod proxy.
//
// This is opt-in. In a default kubeadm cluster kube-controller-manager and
// kube-scheduler listen on 127.0.0.1 only, and their delegated authorizer
// rejects the identity the API server presents when proxying. A job which
// wants these components must start them with --bind-address=0.0.0.0 and add
// the profile path to --authorization-always-allow-paths.
func CheckPods(ctx context.Context, client clientset.Interface, namespaces *regexp.Regexp) []Result {
	nsList, err := client.CoreV1().Namespaces().List(ctx, metav1.ListOptions{})
	if err != nil {
		return []Result{{Component: "pods", Err: fmt.Errorf("listing namespaces: %w", err)}}
	}

	var results []Result
	for _, ns := range nsList.Items {
		if namespaces != nil && !namespaces.MatchString(ns.Name) {
			continue
		}
		pods, err := client.CoreV1().Pods(ns.Name).List(ctx, metav1.ListOptions{})
		if err != nil {
			results = append(results, Result{
				Component: ns.Name,
				Err:       fmt.Errorf("listing pods in %s: %w", ns.Name, err),
			})
			continue
		}
		for _, pod := range pods.Items {
			port, ok := profilePort(&pod)
			if !ok {
				continue
			}
			results = append(results, checkPod(ctx, client, ns.Name, pod.Name, port))
		}
	}
	return results
}

// profilePort returns the secure port to scrape for a known control plane
// component, and whether the pod is one we know how to scrape.
//
// Pod names are matched by prefix because kubeadm appends the node name to
// static pod names. There is no API for identifying control plane components,
// see https://github.com/kubernetes/enhancements/issues/5708.
func profilePort(pod *v1.Pod) (int, bool) {
	switch {
	case strings.HasPrefix(pod.Name, "kube-controller-manager"):
		return 10257, true
	case strings.HasPrefix(pod.Name, "kube-scheduler"):
		return 10259, true
	default:
		return 0, false
	}
}

func checkPod(ctx context.Context, client clientset.Interface, namespace, name string, port int) Result {
	component := namespace + "/" + name

	ctx, cancel := context.WithTimeout(ctx, scrapeTimeout)
	defer cancel()

	body, err := client.CoreV1().RESTClient().Get().
		Namespace(namespace).
		Resource("pods").
		SubResource("proxy").
		Name(fmt.Sprintf("https:%s:%d", name, port)).
		Suffix(profilePath).
		Param("debug", "1").
		DoRaw(ctx)
	if err != nil {
		return Result{Component: component, Err: fmt.Errorf("scraping %s: %w", profilePath, err)}
	}
	res, err := Parse(component, body)
	if err != nil {
		return Result{Component: component, Err: err}
	}
	return res
}

// Report renders results as human readable text. It always lists what was
// checked, including components which reported no leaks, so that a check
// which silently examined nothing is distinguishable from one which passed.
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
