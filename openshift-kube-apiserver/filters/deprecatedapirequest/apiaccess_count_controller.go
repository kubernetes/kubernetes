package deprecatedapirequest

import (
	"context"
	"sync"
	"time"

	apiclientv1 "github.com/openshift/client-go/apiserver/clientset/versioned/typed/apiserver/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/openshift-kube-apiserver/filters/deprecatedapirequest/v1helpers"
)

// NewController returns a controller
func NewController(client apiclientv1.DeprecatedAPIRequestInterface, nodeName string) *controller {
	ret := &controller{
		client:       client,
		nodeName:     nodeName,
		updatePeriod: 5 * time.Minute,
	}
	ret.resetRequestCount()
	return ret
}

// APIRequestLogger support logging deprecated API requests.
type APIRequestLogger interface {
	IsDeprecated(resource, version, group string) bool
	LogRequest(resource schema.GroupVersionResource, timestamp time.Time, user, verb string)
	Start(stop <-chan struct{})
}

type controller struct {
	client       apiclientv1.DeprecatedAPIRequestInterface
	nodeName     string
	updatePeriod time.Duration

	requestCountLock sync.RWMutex
	requestCounts    *apiRequestCounts
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

// LogRequest queues an api request for logging
func (c *controller) LogRequest(resource schema.GroupVersionResource, timestamp time.Time, user, verb string) {
	c.requestCountLock.RLock()
	defer c.requestCountLock.RUnlock()
	c.requestCounts.IncrementRequestCount(resource, timestamp.Hour(), user, verb, 1)
}

// resetCount returns the current count and creates a new requestCount instance var
func (c *controller) resetRequestCount() *apiRequestCounts {
	c.requestCountLock.Lock()
	defer c.requestCountLock.Unlock()
	existing := c.requestCounts
	c.requestCounts = newAPIRequestCounts(c.nodeName)
	return existing
}

// Start the controller
func (c *controller) Start(stop <-chan struct{}) {
	klog.Infof("Starting DeprecatedAPIRequest controller.")

	// create a context.Context needed for some API calls
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-stop
		klog.Infof("Shutting down DeprecatedAPIRequest controller.")
		cancel()
	}()

	// write out logs every c.updatePeriod
	go wait.Until(func() {
		klog.V(2).Infof("updating top APIRequest counts")
		defer klog.V(2).Infof("finished updating top APIRequest counts")

		// get the current count to persist, start a new in-memory count
		countsToPersist := c.resetRequestCount()

		// remove stale data
		expiredHour := (time.Now().Hour() + 1) % 24
		countsToPersist.ExpireOldestCounts(expiredHour)

		// when this function returns, add any remaining counts back to the total to be retried for update
		defer c.requestCounts.Add(countsToPersist)

		var wg sync.WaitGroup
		for _, resourceCount := range countsToPersist.resourceToRequestCount {
			wg.Add(1)
			go func(localResourceCount *resourceRequestCounts) {
				defer wg.Done()

				klog.V(2).Infof("updating top %v APIRequest counts", localResourceCount.resource)
				defer klog.V(2).Infof("finished updating top %v APIRequest counts", localResourceCount.resource)

				_, _, err := v1helpers.ApplyStatus(
					ctx,
					c.client,
					resourceToAPIName(localResourceCount.resource),
					SetRequestCountsForNode(c.nodeName, expiredHour, localResourceCount),
				)
				if err != nil {
					runtime.HandleError(err)
					return
				}

				// on successful update, remove the counts
				countsToPersist.RemoveResource(localResourceCount.resource)
			}(resourceCount)
		}
		wg.Wait()

	}, c.updatePeriod, stop)
}

func resourceToAPIName(resource schema.GroupVersionResource) string {
	apiName := resource.Resource + "." + resource.Version
	if len(resource.Group) > 0 {
		apiName += "." + resource.Group
	}
	return apiName
}
