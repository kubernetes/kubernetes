package deprecatedapirequest

import (
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"

	"github.com/stretchr/testify/assert"
)

func TestRemovedRelease(t *testing.T) {
	rr := removedRelease(
		schema.GroupVersionResource{
			Group:    "flowcontrol.apiserver.k8s.io",
			Version:  "v1alpha1",
			Resource: "flowschemas",
		})
	assert.Equal(t, "1.21", rr)
}

func TestLoggingResetRace(t *testing.T) {
	c := &controller{}
	c.resetRequestCount()

	start := make(chan struct{})
	for i := 0; i < 20; i++ {
		go func() {
			<-start
			for i := 0; i < 100; i++ {
				c.LogRequest(schema.GroupVersionResource{Resource: "pods"}, time.Now(), "user", "verb")
			}
		}()
	}

	for i := 0; i < 10; i++ {
		go func() {
			<-start
			for i := 0; i < 100; i++ {
				c.resetRequestCount()
			}
		}()
	}

	close(start)

	// hope for no data race, which of course failed
}
