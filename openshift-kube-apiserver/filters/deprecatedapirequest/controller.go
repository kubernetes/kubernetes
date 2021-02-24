package deprecatedapirequest

import (
	"context"
	"os"
	"strings"
	"sync"
	"time"

	apiv1 "github.com/openshift/api/apiserver/v1"
	apiclientv1 "github.com/openshift/client-go/apiserver/clientset/versioned/typed/apiserver/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/openshift-kube-apiserver/filters/deprecatedapirequest/v1helpers"
)

// NewController returns a controller
func NewController(config *rest.Config) *controller {
	return &controller{
		client:       apiclientv1.NewForConfigOrDie(configFor(config)).DeprecatedAPIRequests(),
		nodeName:     nodeFor(),
		updatePeriod: 10 * time.Second,
		incoming:     make(chan requestRecord, 100),
	}
}

func configFor(config *rest.Config) *rest.Config {
	c := rest.CopyConfig(config)
	c.AcceptContentTypes = "application/json"
	c.ContentType = "application/json"
	return c
}

func nodeFor() string {
	node := os.Getenv("HOST_IP")
	if hostname, err := os.Hostname(); err != nil {
		node = hostname
	}
	return node
}

// NewPostStartHookFunc returns a PostStartHookFunc that runs after server start.
func NewPostStartHookFunc(controller Controller) server.PostStartHookFunc {
	return func(ctx server.PostStartHookContext) error {
		controller.Start(ctx.StopCh)
		return nil
	}
}

// Controller can be started.
type Controller interface {
	Start(stop <-chan struct{})
}

// APIRequestLogger support logging deprecated API requests.
type APIRequestLogger interface {
	IsDeprecated(resource, version, group string) bool
	LogRequest(resource, version, group, verb, user string, timestamp time.Time)
}

// controller logs deprecated api requests periodically.
// ┌───────────┐   ┌────────────┐                                             ┌──────────┐
// │ServeHTTP()│──▶│ LogRequest │                                             │  Update  │
// └───────────┘   └────────────┘   ┌────────────────────────────────────────▶│  Status  │
//                   │              │                         ┌──────────┐    └──────────┘
//                   │              │  ┌─────────────────────▶│  Update  │          ▲
//                   ▼              │  │          ┌──────────┐│  Status  │          │
//        ╔══════════════╗   ┏━━━━━━━━━━━━━━━┓    │  Update  │└──────────┘          │
//        ║ <-incoming<- ║   ┃requestCounts{}┃───▶│  Status  │     ▲                │
//        ╚══════════════╝   ┗━━━━━━━━━━━━━━━┛    └──────────┘     │                │
//                   │         ▲                        ▲          │                │
//                   ▼         │                        │          │                │
// ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━┳━━━━━━━━━━┓
// ┃   Start()   ┃   go func1()    ┃ go func2()   │resource_1│resource_2│ ... │resource_n┃
// ┗━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━┻━━━━━━━━━━┻━━━━━━━━━━┻━━━━━┻━━━━━━━━━━┛
type controller struct {
	client       apiclientv1.DeprecatedAPIRequestInterface
	nodeName     string
	updatePeriod time.Duration
	incoming     chan requestRecord
}

// requestRecord represents an api resource request to be logged
type requestRecord struct {
	resource  string
	user      string
	verb      string
	timestamp time.Time
}

// IsDeprecated return true if the resource is deprecated.
func (c *controller) IsDeprecated(resource, version, group string) bool {
	_, ok := deprecatedApiRemovedRelease[schema.GroupVersionResource{
		Group:    group,
		Version:  version,
		Resource: resource,
	}]
	return ok
}

// removedRelease of a specified resource.version.group.
func (c *controller) removedRelease(r string) string {
	s := strings.SplitN(r, ".", 3)
	gvr := schema.GroupVersionResource{Resource: s[0], Version: s[1]}
	if len(s) == 3 {
		gvr.Group = s[2]
	}
	return deprecatedApiRemovedRelease[gvr]
}

// LogRequest queues an api request for logging
func (c *controller) LogRequest(resource, version, group, verb, user string, ts time.Time) {
	r := resource + "." + version
	if len(group) > 0 {
		r += "." + group
	}
	select {
	case c.incoming <- requestRecord{resource: r, user: user, verb: verb, timestamp: ts}:
	default:
		// oh, well. best effort (don't want to hang up the http request chain).
	}
}

// Start the controller
func (c *controller) Start(stop <-chan struct{}) {
	// create a context.Context needed for some API calls
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-stop
		klog.V(1).Infof("Shutting down deprecatedapirequest controller.")
		cancel()
	}()

	// requestCounts is the source of truth for the current node.
	requestCounts := &apiRequestCounts{resources: map[string]*resourceRequestCounts{}}

	// start goroutine that adds incoming log records to the in-memory logs
	go func() {
		loaded := sets.NewString()
		for {
			select {
			case rr := <-c.incoming:
				if !loaded.Has(rr.resource) {
					// load existing persisted counts from a previous run
					previouslyPersisted, err := c.loadOrCreateLogForResource(ctx, rr.resource)
					if err != nil {
						// something went wrong, we'll try again another next time
						runtime.HandleError(err)
						continue
					}
					// add previously persisted counts
					requestCounts.Resource(rr.resource).IncrementRequestCounts(previouslyPersisted)
					// remember we've already loaded previously persisted counts for this resource
					loaded.Insert(rr.resource)
				}
				// add log record to in-memory log
				requestCounts.IncrementRequestCount(rr.resource, rr.timestamp, rr.user, rr.verb, 1)
			case <-stop:
				return
			}
		}
	}()

	// every hour, expire old counts
	go wait.Until(func() {
		requestCounts.ExpireOldestCounts(time.Now())
	}, 1*time.Hour, stop)

	// write out logs every c.updatePeriod
	go wait.Until(func() {
		var wg sync.WaitGroup
		for resource, resources := range requestCounts.resources {
			go func(resource string, resources *resourceRequestCounts) {
				wg.Add(1)
				defer wg.Done()
				_, _, err := v1helpers.UpdateStatus(ctx, c.client, resource, SetRequestCountsForNode(c.nodeName, resources))
				if err != nil {
					runtime.HandleError(err)
					return
				}
			}(resource, resources)
		}
		wg.Wait()
	}, c.updatePeriod, stop)
}

func (c *controller) loadOrCreateLogForResource(ctx context.Context, resource string) (*resourceRequestCounts, error) {
	log, err := c.getOrCreateLogForResource(ctx, resource)
	requestCounts := &apiRequestCounts{resources: map[string]*resourceRequestCounts{}}
	if err != nil {
		return nil, err
	}
	for _, requestLog := range log.Status.RequestsLast24h {
		for _, nodeRequestLog := range requestLog.Nodes {
			if nodeRequestLog.NodeName != c.nodeName {
				continue
			}
			timestamp := nodeRequestLog.LastUpdate.Time
			for _, requestUser := range nodeRequestLog.Users {
				user := requestUser.UserName
				for _, requestCount := range requestUser.Requests {
					verb := requestCount.Verb
					count := requestCount.Count
					requestCounts.IncrementRequestCount(resource, timestamp, user, verb, count)
				}
			}
		}
	}
	return requestCounts.Resource(resource), nil
}

func (c *controller) getOrCreateLogForResource(ctx context.Context, resource string) (*apiv1.DeprecatedAPIRequest, error) {
	log, err := c.client.Get(ctx, resource, metav1.GetOptions{})
	if err == nil {
		return log, err
	}
	if !errors.IsNotFound(err) {
		return nil, err
	}
	return c.client.Create(ctx, &apiv1.DeprecatedAPIRequest{
		ObjectMeta: metav1.ObjectMeta{Name: resource},
		Spec:       apiv1.DeprecatedAPIRequestSpec{RemovedRelease: c.removedRelease(resource)},
	}, metav1.CreateOptions{})
}
