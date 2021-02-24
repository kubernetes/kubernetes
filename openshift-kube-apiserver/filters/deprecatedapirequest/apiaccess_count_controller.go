package deprecatedapirequest

import (
	"context"
	"math/rand"
	"strings"
	"sync"
	"time"

	apiv1 "github.com/openshift/api/apiserver/v1"

	apiclientv1 "github.com/openshift/client-go/apiserver/clientset/versioned/typed/apiserver/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/openshift-kube-apiserver/filters/deprecatedapirequest/v1helpers"
)

// NewController returns a controller
func NewController(client apiclientv1.APIRequestCountInterface, nodeName string) *controller {
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
	LogRequest(resource schema.GroupVersionResource, timestamp time.Time, user, userAgent, verb string)
	Start(stop <-chan struct{})
}

type controller struct {
	client       apiclientv1.APIRequestCountInterface
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
func (c *controller) LogRequest(resource schema.GroupVersionResource, timestamp time.Time, user, userAgent, verb string) {
	c.requestCountLock.RLock()
	defer c.requestCountLock.RUnlock()
	// we snip user agents to reduce cardinality and unique keys.  For well behaved agents, we see useragents about like
	// kube-controller-manager/v1.21.0 (linux/amd64) kubernetes/743bd58/kube-controller-manager
	// so we will snip at the first space.
	snippedUserAgent := userAgent
	if i := strings.Index(userAgent, " "); i > 0 {
		snippedUserAgent = userAgent[:i]
	}
	userKey := userKey{
		user:      user,
		userAgent: snippedUserAgent,
	}
	c.requestCounts.IncrementRequestCount(resource, timestamp.Hour(), userKey, verb, 1)
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
	go wait.NonSlidingUntilWithContext(ctx, c.persistRequestCountForAllResources, c.updatePeriod)
}

func (c *controller) persistRequestCountForAllResources(ctx context.Context) {
	klog.V(2).Infof("updating top APIRequest counts")
	defer klog.V(2).Infof("finished updating top APIRequest counts")

	// get the current count to persist, start a new in-memory count
	countsToPersist := c.resetRequestCount()

	// remove stale data
	expiredHour := (time.Now().Hour() + 1) % 24
	currentHour := time.Now().Hour()
	countsToPersist.ExpireOldestCounts(expiredHour)

	// when this function returns, add any remaining counts back to the total to be retried for update
	defer c.requestCounts.Add(countsToPersist)

	var wg sync.WaitGroup
	for gvr := range countsToPersist.resourceToRequestCount {
		resourceCount := countsToPersist.Resource(gvr)
		wg.Add(1)
		go func() {
			time.Sleep(time.Duration(rand.Int63n(int64(c.updatePeriod / 5 * 4)))) // smear out over the interval to avoid resource spikes
			c.persistRequestCountForResource(ctx, &wg, currentHour, expiredHour, resourceCount)
		}()
	}
	wg.Wait()
}

func (c *controller) persistRequestCountForResource(ctx context.Context, wg *sync.WaitGroup, currentHour, expiredHour int, localResourceCount *resourceRequestCounts) {
	defer wg.Done()

	klog.V(2).Infof("updating top %v APIRequest counts", localResourceCount.resource)
	defer klog.V(2).Infof("finished updating top %v APIRequest counts", localResourceCount.resource)

	status, _, err := v1helpers.ApplyStatus(
		ctx,
		c.client,
		resourceToAPIName(localResourceCount.resource),
		SetRequestCountsForNode(c.nodeName, currentHour, expiredHour, localResourceCount),
	)
	if err != nil {
		runtime.HandleError(err)
		return
	}

	// on successful update, remove the counts we don't need.  This is every hour except the current hour
	// and every user recorded for the current hour on this node
	removePersistedRequestCounts(c.nodeName, currentHour, status, localResourceCount)
}

// removePersistedRequestCounts removes the counts we don't need to keep in memory.
// This is every hour except the current hour (those will no longer change) and every user recorded for the current hour on this node.
// Then it tracks the amount that needs to be kept out of the sum. This is logically the amount we're adding back in.
// Because we already counted all the users in the persisted sum, we need to exclude the amount we'll be placing back
// in memory.
func removePersistedRequestCounts(nodeName string, currentHour int, persistedStatus *apiv1.APIRequestCountStatus, localResourceCount *resourceRequestCounts) {
	for hourIndex := range localResourceCount.hourToRequestCount {
		if currentHour != hourIndex {
			localResourceCount.RemoveHour(hourIndex)
		}
	}
	for _, persistedNodeCount := range persistedStatus.CurrentHour.ByNode {
		if persistedNodeCount.NodeName != nodeName {
			continue
		}
		for _, peristedUserCount := range persistedNodeCount.ByUser {
			userKey := userKey{
				user:      peristedUserCount.UserName,
				userAgent: peristedUserCount.UserAgent,
			}
			localResourceCount.Hour(currentHour).RemoveUser(userKey)
		}
	}

	countToSuppress := int64(0)
	for _, userCounts := range localResourceCount.Hour(currentHour).usersToRequestCounts {
		for _, verbCount := range userCounts.verbsToRequestCounts {
			countToSuppress += verbCount.count
		}
	}

	localResourceCount.Hour(currentHour).countToSuppress = countToSuppress
}

func resourceToAPIName(resource schema.GroupVersionResource) string {
	apiName := resource.Resource + "." + resource.Version
	if len(resource.Group) > 0 {
		apiName += "." + resource.Group
	}
	return apiName
}
