package cache

import (
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	fcache "k8s.io/client-go/tools/cache/testing"

	"errors"
	"fmt"
	"testing"
	"time"
)

const (
	IndexApp = "appLabel"
)

var appIndex = Indexers{IndexApp: func(obj interface{}) ([]string, error) {
	metaData, err := meta.Accessor(obj)
	if err != nil {
		return []string{""}, fmt.Errorf("object has no meta: %v", err)
	}

	labels := metaData.GetLabels()
	if _, ok := labels["app"]; !ok {
		return []string{"unknown"}, nil
	}

	return []string{labels["app"]}, nil
}}

func fakeInformer() *sharedIndexInformer {
	// fake controller source
	source := fcache.NewFakeControllerSource()
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "h-dev", Labels: map[string]string{"app": "hello", "env": "dev"}}})
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "h-prod", Labels: map[string]string{"app": "hello", "env": "prod"}}})
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "h2-dev", Labels: map[string]string{"app": "hello2", "env": "dev"}}})
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "h2-prod", Labels: map[string]string{"app": "hello2", "env": "prod"}}})

	// create the shared informer and resync every 1s
	informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second).(*sharedIndexInformer)
	_ = informer.AddIndexers(appIndex)
	return informer
}

type args struct {
	indexer  Indexer
	name     string
	key      string
	selector labels.Selector
	ret      []*v1.Pod
}

func (l *args) appendFunc(m interface{}) {
	l.ret = append(l.ret, m.(*v1.Pod))
}

func (l *args) matchResult(resources map[string]bool) error {

	for _, r := range l.ret {
		if _, ok := resources[r.Name]; !ok {
			return errors.New(fmt.Sprintf("test error,should not find item: %s", r.Name))
		}
		resources[r.Name] = true
	}

	if len(resources) != len(l.ret) {
		return errors.New(fmt.Sprintf("test found not match %d, %d", len(resources), len(l.ret)))
	}

	for k, v := range resources {
		if !v {
			return errors.New(fmt.Sprintf(" test not found %s", k))
		}
	}
	return nil
}

func TestListByIndex(t *testing.T) {

	// prepare base env
	informer := fakeInformer()
	stop := make(chan struct{})
	defer close(stop)
	go informer.Run(stop)
	WaitForCacheSync(stop, informer.HasSynced)

	devEnvSelector, _ := labels.Parse("env=dev")
	tests := []struct {
		name      string
		args      args
		resources map[string]bool
	}{
		{
			name: "hello-dev",
			args: args{
				name:     IndexApp,
				key:      "hello",
				selector: devEnvSelector,
				indexer:  informer.GetIndexer(),
			},
			resources: map[string]bool{"h-dev": false},
		},
		{
			name: "hello2-dev",
			args: args{
				name:     IndexApp,
				key:      "hello2",
				selector: devEnvSelector,
				indexer:  informer.GetIndexer(),
			},
			resources: map[string]bool{"h2-dev": false},
		},
		{
			name: "hello3-dev",
			args: args{
				name:     IndexApp,
				key:      "hello3",
				selector: devEnvSelector,
				indexer:  informer.GetIndexer(),
			},
			resources: map[string]bool{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ListByIndex(tt.args.indexer, tt.args.name, tt.args.key, tt.args.selector, tt.args.appendFunc); err != nil {
				t.Fatalf("ListByIndex() error = %v", err)
			}

			if err := tt.args.matchResult(tt.resources); err != nil {
				t.Fatalf("ListByIndex() error = %v", err)
			}

		})
	}
}
