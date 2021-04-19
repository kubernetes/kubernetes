/*
Copyright 2014 The Kubernetes Authors.

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

package watch

import (
	"sync"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// FullChannelBehavior controls how the Broadcaster reacts if a watcher's watch
// channel is full.
type FullChannelBehavior int

const (
	WaitIfChannelFull FullChannelBehavior = iota
	DropIfChannelFull
)

// Buffer the incoming queue a little bit even though it should rarely ever accumulate
// anything, just in case a few events are received in such a short window that
// Broadcaster can't move them onto the watchers' queues fast enough.
const incomingQueueLength = 25

// Broadcaster distributes event notifications among any number of watchers. Every event
// is delivered to every watcher.
type Broadcaster struct {
	watchers     map[int64]*broadcasterWatcher
	//新加入watch 的ID
	nextWatcher  int64
	 //创建的时候add 关闭发送的时候done， 退出的时候wait 等所有的watch 都关闭了
	distributing sync.WaitGroup

	 //event 队列， 从队列取出event 发送到watchers
	incoming chan Event
	stopped  chan struct{}

	// How large to make watcher's channel.
	 // watch 队列的长度，给新建watch 用的
	watchQueueLength int
	// If one of the watch channels is full, don't wait for it to become empty.
	// Instead just deliver it to the watchers that do have space in their
	// channels and move on to the next event.
	// It's more fair to do this on a per-watcher basis than to do it on the
	// "incoming" channel, which would allow one slow watcher to prevent all
	// other watchers from getting new events.
	 // 分发时阻塞或者非阻塞
	fullChannelBehavior FullChannelBehavior
}

// queueLength 是watch 的长度 ，广播的队列的长度是25
// 对外提供Watch方法，添加watch 和 Action & ActionOrDrop 发送event
// NewBroadcaster creates a new Broadcaster. queueLength is the maximum number of events to queue per watcher.
// It is guaranteed that events will be distributed in the order in which they occur,
// but the order in which a single event is distributed among all of the watchers is unspecified.
func NewBroadcaster(queueLength int, fullChannelBehavior FullChannelBehavior) *Broadcaster {
	/*
	  创建一个broadcaster， 通过incoming channel并向所有的 watchers 发送 event， 默认队列的长度是25.
	*/
	m := &Broadcaster{
		 // 初始化的watchers 是空的， 这个怎么加进去呢？
		watchers:            map[int64]*broadcasterWatcher{},
		incoming:            make(chan Event, incomingQueueLength),
		stopped:             make(chan struct{}),
		watchQueueLength:    queueLength,
		 // 当某个watcher队列满了 通过这个参数控制跳过还是阻塞
		fullChannelBehavior: fullChannelBehavior,
	}
	m.distributing.Add(1)
	 // 后台发送event
	go m.loop()
	return m
}

// queueLength 是广播和watch 的长度 2个设成一样
// NewLongQueueBroadcaster functions nearly identically to NewBroadcaster,
// except that the incoming queue is the same size as the outgoing queues
// (specified by queueLength).
func NewLongQueueBroadcaster(queueLength int, fullChannelBehavior FullChannelBehavior) *Broadcaster {
	// 跟 NewBroadcaster 一样，除了队列长度
	m := &Broadcaster{
		watchers:            map[int64]*broadcasterWatcher{},
		incoming:            make(chan Event, queueLength),
		stopped:             make(chan struct{}),
		watchQueueLength:    queueLength,
		fullChannelBehavior: fullChannelBehavior,
	}
	m.distributing.Add(1)
	go m.loop()
	return m
}

const internalRunFunctionMarker = "internal-do-function"

// a function type we can shoehorn into the queue.
 // 这个dummy event是为了在队列里添加 watch或者stop watch action
type functionFakeRuntimeObject func()

func (obj functionFakeRuntimeObject) GetObjectKind() schema.ObjectKind {
	return schema.EmptyObjectKind
}
func (obj functionFakeRuntimeObject) DeepCopyObject() runtime.Object {
	if obj == nil {
		return nil
	}
	// funcs are immutable. Hence, just return the original func.
	return obj
}

// Execute f, blocking the incoming queue (and waiting for it to drain first).
// The purpose of this terrible hack is so that watchers added after an event
// won't ever see that event, and will always see any event after they are
// added.
func (m *Broadcaster) blockQueue(f func()) {
	select {
	case <-m.stopped:
		 // 如果stop 了， 立刻返回，并在155行panic
		return
	default:
	}
	var wg sync.WaitGroup
	wg.Add(1)
	m.incoming <- Event{
		Type: internalRunFunctionMarker,
		Object: functionFakeRuntimeObject(func() {
			defer wg.Done()
			 // 添加的f 在这里没有被执行，在253行才执行
			f()
		}),
	}
	 // 等添加watch 的action结束， 要不159行会panic
	wg.Wait()
}

// 这里的总体思想是把添加watch的动作放到队列里，在分发之前的时候做添加watch的动作，
// 因为map是进程不安全的，直接强行加进去 后台的分发如果还没执行完毕会有问题
// watch 后 返回result channel （得到分发结果）和stop channel （关闭watch）
// Watch adds a new watcher to the list and returns an Interface for it.
// Note: new watchers will only receive new events. They won't get an entire history
// of previous events. It will block until the watcher is actually added to the
// broadcaster.
func (m *Broadcaster) Watch() Interface {
	var w *broadcasterWatcher
	 // 添加watch这个func 添加到队列里，在253行的loop函数处理event的时候调用
	m.blockQueue(func() {
		//为什么不直接添加，而是等event 处理的时候才调用？ 因为map是进程不安全的，直接强行加进去 后台的分发如果还没执行完毕会有问题
		id := m.nextWatcher
		m.nextWatcher++
		w = &broadcasterWatcher{
			result:  make(chan Event, m.watchQueueLength),
			stopped: make(chan struct{}),
			id:      id,
			// watch 要 Broadcaster 字段 有什么用?
			//watch 可以通过这个字段的调用Broadcaster的stop watch 方法，通过id 把自己移除watch 列表
			m:       m,
		}
		m.watchers[id] = w
		// 这个function 被传进去 但是还没执行，w是nil
	})
	if w == nil {
		/* w 才253行执行的时候才真正不是nil，
		如果通过blockQueue的event在队列里被调用慢了，这里会不会panic
		答案是不会，因为132行wait了，就是要等这个函数执行完，并且w 真正赋值了
		*/
		// The panic here is to be consistent with the previous interface behavior
		// we are willing to re-evaluate in the future.
		panic("broadcaster already stopped")
	}
	return w
}

// 添加到分发队列之前， 往自己的result channel 写一些event， 只发给自己这个watch，其他的watch 不会收到这些event
// WatchWithPrefix adds a new watcher to the list and returns an Interface for it. It sends
// queuedEvents down the new watch before beginning to send ordinary events from Broadcaster.
// The returned watch will have a queue length that is at least large enough to accommodate
// all of the items in queuedEvents. It will block until the watcher is actually added to
// the broadcaster.
func (m *Broadcaster) WatchWithPrefix(queuedEvents []Event) Interface {
	var w *broadcasterWatcher
	 // watch
	m.blockQueue(func() {
		id := m.nextWatcher
		m.nextWatcher++
		length := m.watchQueueLength
		if n := len(queuedEvents) + 1; n > length {
			length = n
		}
		w = &broadcasterWatcher{
			result:  make(chan Event, length),
			stopped: make(chan struct{}),
			id:      id,
			m:       m,
		}
		m.watchers[id] = w
		for _, e := range queuedEvents {
			 // 添加到分发队列之前， 往自己的result channel 写一些event
			w.result <- e
		}
	})
	if w == nil {
		// The panic here is to be consistent with the previous interface behavior
		// we are willing to re-evaluate in the future.
		panic("broadcaster already stopped")
	}
	return w
}

// stopWatching stops the given watcher and removes it from the list.
func (m *Broadcaster) stopWatching(id int64) {
	m.blockQueue(func() {
		w, ok := m.watchers[id]
		if !ok {
			// No need to do anything, it's already been removed from the list.
			return
		}
		delete(m.watchers, id)
		close(w.result)
	})
}

// closeAll disconnects all watchers (presumably in response to a Shutdown call).
func (m *Broadcaster) closeAll() {
	for _, w := range m.watchers {
		close(w.result)
	}
	// Delete everything from the map, since presence/absence in the map is used
	// by stopWatching to avoid double-closing the channel.
	m.watchers = map[int64]*broadcasterWatcher{}
}

// Action 向Broadcaster 里的所有watch 发送 event
// Action distributes the given event among all watchers.
func (m *Broadcaster) Action(action EventType, obj runtime.Object) {
	m.incoming <- Event{action, obj}
}

// 像广播队列里发送 event， 如果队列满了 发送失败返回false
// Action distributes the given event among all watchers, or drops it on the floor
// if too many incoming actions are queued up.  Returns true if the action was sent,
// false if dropped.
func (m *Broadcaster) ActionOrDrop(action EventType, obj runtime.Object) bool {
	select {
	case m.incoming <- Event{action, obj}:
		return true
	default:
		return false
	}
}

// Shutdown disconnects all watchers (but any queued events will still be distributed).
// You must not call Action or Watch* after calling Shutdown. This call blocks
// until all events have been distributed through the outbound channels. Note
// that since they can be buffered, this means that the watchers might not
// have received the data yet as it can remain sitting in the buffered
// channel. It will block until the broadcaster stop request is actually executed
func (m *Broadcaster) Shutdown() {
	m.blockQueue(func() {
		  //同理 在分发执行前关闭
		close(m.stopped)
		//关闭后不分发了 line260
		close(m.incoming)
	})
	 // 等关闭所有watch
	m.distributing.Wait()
}

// loop receives from m.incoming and distributes to all watchers.
func (m *Broadcaster) loop() {
	// Deliberately not catching crashes here. Yes, bring down the process if there's a
	// bug in watch.Broadcaster.
	{ // 通过channel 不停的读取event，并通过Shutdown 关闭
	for event := range m.incoming {
		if event.Type == internalRunFunctionMarker {
			/* 类型断言执行添加watcher
			   这个不是真正的event 只是为了在队列里添加watch，所以跳过分发步骤
			*/
			 // 类型断言执行添加watcher
			event.Object.(functionFakeRuntimeObject)()
			continue
		}
		 // 往watch map的watch 分发event
		m.distribute(event)
	}
	// 关闭所有watch
	m.closeAll()
	m.distributing.Done()
}

// distribute sends event to all watchers. Blocking.
func (m *Broadcaster) distribute(event Event) {
	if m.fullChannelBehavior == DropIfChannelFull {
		for _, w := range m.watchers {
			select {
				 // 非阻塞模式，有的watch队列满了，跳过这个watch 继续下一个watch
			case w.result <- event:
				 // watch通道关闭了 就不写了， watch 方法返回的stop 函数会关闭watch，line 312
			case <-w.stopped:
			default: // Don't block if the event can't be queued.
			}
		}
	} else {
		for _, w := range m.watchers {
			select {
				 // 阻塞模式， 等这个watch 有buffer了，才继续
			case w.result <- event:
			case <-w.stopped:
			}
		}
	}
}

// broadcasterWatcher handles a single watcher of a broadcaster
type broadcasterWatcher struct {
	result  chan Event
	stopped chan struct{}
	stop    sync.Once
	id      int64
	 // 把通道放进来， 关闭的时候通过他的stopWatching方法移除自己
	m       *Broadcaster
}

// ResultChan returns a channel to use for waiting on events.
func (mw *broadcasterWatcher) ResultChan() <-chan Event {
	//返回 event channel
	return mw.result
}

// Stop stops watching and removes mw from its list.
// It will block until the watcher stop request is actually executed
func (mw *broadcasterWatcher) Stop() {
	mw.stop.Do(func() {
		// 关闭watch
		close(mw.stopped)
		 //把自己移出Broadcaster的watch列表
		mw.m.stopWatching(mw.id)
	})
}
