package v2

import (
	"context"
	"errors"
	"net/http"
	"reflect"
	"sync"
	"time"

	"github.com/emicklei/go-restful/v3"
	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
)

type FakeResourceManager interface {
	ResourceManager
	Expect() ResourceManager

	HasExpectedNumberActions() bool
	Validate() error
	WaitForActions(ctx context.Context, timeout time.Duration) error
}

func NewFakeResourceManager() FakeResourceManager {
	return &fakeResourceManager{}
}

// a resource manager with helper functions for checking the actions
// match expected. For Use in tests
type fakeResourceManager struct {
	recorderResourceManager
	expect recorderResourceManager
}

// a resource manager which instead of managing a discovery document,
// simply records the calls to its interface functoins for testing
type recorderResourceManager struct {
	lock    sync.RWMutex
	Actions []recorderResourceManagerAction
}

var _ ResourceManager = &fakeResourceManager{}
var _ ResourceManager = &recorderResourceManager{}

// Storage type for a call to the resource manager
type recorderResourceManagerAction struct {
	Type    string
	Group   string
	Version string
	Value   interface{}
}

func (f *fakeResourceManager) Expect() ResourceManager {
	return &f.recorderResourceManager
}

func (f *fakeResourceManager) HasExpectedNumberActions() bool {
	f.lock.RLock()
	defer f.lock.RUnlock()

	f.expect.lock.RLock()
	defer f.expect.lock.RUnlock()

	return len(f.Actions) >= len(f.expect.Actions)
}

func (f *fakeResourceManager) Validate() error {
	f.lock.RLock()
	defer f.lock.RUnlock()

	f.expect.lock.RLock()
	defer f.expect.lock.RUnlock()

	if !reflect.DeepEqual(f.expect.Actions, f.Actions) {
		return errors.New(cmp.Diff(f.expect.Actions, f.Actions))
	}
	return nil
}

func (f *fakeResourceManager) WaitForActions(ctx context.Context, timeout time.Duration) error {
	err := wait.PollImmediateWithContext(
		ctx,
		100*time.Millisecond, // try every 100ms
		1*time.Second,        // timeout after 1s
		func(ctx context.Context) (done bool, err error) {
			if f.HasExpectedNumberActions() {
				return true, f.Validate()
			}
			return false, nil
		})
	return err
}

func (f *recorderResourceManager) AddGroupVersion(groupName string, value metav1.APIVersionDiscovery) {
	f.lock.Lock()
	defer f.lock.Unlock()

	f.Actions = append(f.Actions, recorderResourceManagerAction{
		Type:  "AddGroupVersion",
		Group: groupName,
		Value: value,
	})
}
func (f *recorderResourceManager) RemoveGroup(groupName string) {
	f.lock.Lock()
	defer f.lock.Unlock()

	f.Actions = append(f.Actions, recorderResourceManagerAction{
		Type:  "RemoveGroup",
		Group: groupName,
	})

}
func (f *recorderResourceManager) RemoveGroupVersion(gv metav1.GroupVersion) {
	f.lock.Lock()
	defer f.lock.Unlock()

	f.Actions = append(f.Actions, recorderResourceManagerAction{
		Type:    "RemoveGroupVersion",
		Group:   gv.Group,
		Version: gv.Version,
	})

}
func (f *recorderResourceManager) SetGroups(values []metav1.APIGroupDiscovery) {
	f.lock.Lock()
	defer f.lock.Unlock()

	f.Actions = append(f.Actions, recorderResourceManagerAction{
		Type:  "SetGroups",
		Value: values,
	})
}
func (f *recorderResourceManager) WebService() *restful.WebService {
	panic("unimplemented")
}

func (f *recorderResourceManager) ServeHTTP(http.ResponseWriter, *http.Request) {
	panic("unimplemented")
}
