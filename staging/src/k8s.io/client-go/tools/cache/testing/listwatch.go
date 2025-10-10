package framework

import (
	"k8s.io/client-go/tools/cache"
)

type listWatcherWithUnSupportedWatchListSemanticsWrapper struct {
	cache.ListerWatcher
}

func (lw listWatcherWithUnSupportedWatchListSemanticsWrapper) IsWatchListSemanticsUnSupported() bool {
	return true
}

func ToListWatcherWithUnSupportedWatchListSemantics(lw cache.ListerWatcher) cache.ListerWatcher {
	return listWatcherWithUnSupportedWatchListSemanticsWrapper{
		lw,
	}
}
