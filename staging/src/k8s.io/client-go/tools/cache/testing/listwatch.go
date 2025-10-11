package framework

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
)

type Lister interface {
	List(options metav1.ListOptions) (runtime.Object, error)
}

type Watcher interface {
	Watch(options metav1.ListOptions) (watch.Interface, error)
}

type ListerWatcher interface {
	Lister
	Watcher
}

type listWatcherWithUnSupportedWatchListSemanticsWrapper struct {
	ListerWatcher
}

func (lw listWatcherWithUnSupportedWatchListSemanticsWrapper) IsWatchListSemanticsUnSupported() bool {
	return true
}

func ToListWatcherWithUnSupportedWatchListSemantics(lw ListerWatcher) ListerWatcher {
	return listWatcherWithUnSupportedWatchListSemanticsWrapper{
		lw,
	}
}
