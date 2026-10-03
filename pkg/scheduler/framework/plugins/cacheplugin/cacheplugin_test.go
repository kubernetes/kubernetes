/*
Copyright 2025 The Kubernetes Authors.

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

package cacheplugin

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

// mockCacheForPlugin 是 CacheForPlugin 接口的模拟实现
type mockCacheForPlugin struct {
	processReservePodCalls   []reservePodCall
	processUnreservePodCalls []unreservePodCall
	lock                     sync.Mutex
}

type reservePodCall struct {
	pod      *corev1.Pod
	nodeName string
}

type unreservePodCall struct {
	pod      *corev1.Pod
	nodeName string
}

func (m *mockCacheForPlugin) ProcessReservePod(new *corev1.Pod, reservedNode string) {
	m.lock.Lock()
	defer m.lock.Unlock()
	m.processReservePodCalls = append(m.processReservePodCalls, reservePodCall{
		pod:      new,
		nodeName: reservedNode,
	})
}

func (m *mockCacheForPlugin) ProcessUnreservePod(new *corev1.Pod, reservedNode string) {
	m.lock.Lock()
	defer m.lock.Unlock()
	m.processUnreservePodCalls = append(m.processUnreservePodCalls, unreservePodCall{
		pod:      new,
		nodeName: reservedNode,
	})
}

func (m *mockCacheForPlugin) GetProcessReservePodCalls() []reservePodCall {
	m.lock.Lock()
	defer m.lock.Unlock()
	calls := make([]reservePodCall, len(m.processReservePodCalls))
	copy(calls, m.processReservePodCalls)
	return calls
}

func (m *mockCacheForPlugin) GetProcessUnreservePodCalls() []unreservePodCall {
	m.lock.Lock()
	defer m.lock.Unlock()
	calls := make([]unreservePodCall, len(m.processUnreservePodCalls))
	copy(calls, m.processUnreservePodCalls)
	return calls
}

func TestCachePlugin_Name(t *testing.T) {
	t.Run("测试插件名称", func(t *testing.T) {
		plugin := &CachePlugin{
			caches: make([]CacheForPlugin, 0),
		}

		name := plugin.Name()
		expectedName := Name

		if name != expectedName {
			t.Errorf("Expected name %s, got %s", expectedName, name)
		}
	})
}

func TestCachePlugin_Reserve(t *testing.T) {
	t.Run("测试Reserve方法", func(t *testing.T) {
		// 创建模拟缓存
		mockCache1 := &mockCacheForPlugin{}
		mockCache2 := &mockCacheForPlugin{}

		// 创建插件并添加模拟缓存
		plugin := &CachePlugin{
			caches: []CacheForPlugin{mockCache1, mockCache2},
		}

		// 创建测试Pod和节点名，使用测试工具
		pod := st.MakePod().Name("test-pod").UID("test-pod").Namespace("default").Node("test-node").Obj()
		nodeName := "test-node"

		// 调用Reserve方法
		status := plugin.Reserve(context.Background(), &framework.CycleState{}, pod, nodeName)

		// 验证返回状态
		if status != nil {
			t.Errorf("Expected nil status, got %v", status)
		}

		// 等待一段时间确保所有调用都已完成
		time.Sleep(10 * time.Millisecond)

		// 验证所有缓存的ProcessReservePod方法都被调用
		for i, mockCache := range []*mockCacheForPlugin{mockCache1, mockCache2} {
			calls := mockCache.GetProcessReservePodCalls()
			if len(calls) != 1 {
				t.Errorf("Expected 1 call to ProcessReservePod for cache %d, got %d", i+1, len(calls))
				continue
			}

			if calls[0].pod != pod {
				t.Errorf("Expected pod %v, got %v", pod, calls[0].pod)
			}

			if calls[0].nodeName != nodeName {
				t.Errorf("Expected node name %s, got %s", nodeName, calls[0].nodeName)
			}
		}
	})
}

func TestCachePlugin_Unreserve(t *testing.T) {
	t.Run("测试Unreserve方法", func(t *testing.T) {
		// 创建模拟缓存
		mockCache1 := &mockCacheForPlugin{}
		mockCache2 := &mockCacheForPlugin{}

		// 创建插件并添加模拟缓存
		plugin := &CachePlugin{
			caches: []CacheForPlugin{mockCache1, mockCache2},
		}

		// 创建测试Pod和节点名，使用测试工具
		pod := st.MakePod().Name("test-pod").UID("test-pod").Namespace("default").Node("test-node").Obj()
		nodeName := "test-node"

		// 调用Unreserve方法
		plugin.Unreserve(context.Background(), &framework.CycleState{}, pod, nodeName)

		// 等待一段时间确保所有调用都已完成
		time.Sleep(10 * time.Millisecond)

		// 验证所有缓存的ProcessUnreservePod方法都被调用
		for i, mockCache := range []*mockCacheForPlugin{mockCache1, mockCache2} {
			calls := mockCache.GetProcessUnreservePodCalls()
			if len(calls) != 1 {
				t.Errorf("Expected 1 call to ProcessUnreservePod for cache %d, got %d", i+1, len(calls))
				continue
			}

			if calls[0].pod != pod {
				t.Errorf("Expected pod %v, got %v", pod, calls[0].pod)
			}

			if calls[0].nodeName != nodeName {
				t.Errorf("Expected node name %s, got %s", nodeName, calls[0].nodeName)
			}
		}
	})
}

func TestNew(t *testing.T) {
	t.Run("测试New函数", func(t *testing.T) {
		// 调用New函数
		plugin, err := New(context.Background(), nil, nil)

		// 验证没有错误
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}

		// 验证返回了插件实例
		if plugin == nil {
			t.Error("Expected plugin instance, got nil")
		}

		// 验证返回的是全局cacheplugin实例
		if plugin != cacheplugin {
			t.Error("Expected to return global cacheplugin instance")
		}
	})
}

// 测试CacheImpl相关功能
func TestCacheImpl_AddIfNotPresent(t *testing.T) {
	t.Run("测试AddIfNotPresent方法", func(t *testing.T) {
		// 创建测试缓存
		keyFunc := func(s string) string { return s }
		cache := &CacheImpl[string, int]{
			Name:          "test-cache",
			Size:          3,
			PriorityQueue: make([]string, 0),
			HashMap:       make(map[string]*ItemInfo[int]),
			OriginMap:     make(map[string]string),
			KeyFunc:       keyFunc,
			logger:        logr.Discard(),
		}

		// 添加项目
		cache.AddIfNotPresent("key1", 100)
		cache.AddIfNotPresent("key2", 200)

		// 验证项目已添加
		if len(cache.PriorityQueue) != 2 {
			t.Errorf("Expected queue length 2, got %d", len(cache.PriorityQueue))
		}

		if len(cache.HashMap) != 2 {
			t.Errorf("Expected hash map length 2, got %d", len(cache.HashMap))
		}

		if len(cache.OriginMap) != 2 {
			t.Errorf("Expected origin map length 2, got %d", len(cache.OriginMap))
		}

		// 尝试再次添加相同键的项目，应该被忽略
		cache.AddIfNotPresent("key1", 300)

		// 验证项目未被更新
		if cache.HashMap["key1"].Item != 100 {
			t.Errorf("Expected item value 100, got %d", cache.HashMap["key1"].Item)
		}

		// 添加超过缓存大小的项目
		cache.AddIfNotPresent("key3", 300)
		cache.AddIfNotPresent("key4", 400) // 这应该导致最早添加的key1被删除

		// 验证缓存大小限制
		if len(cache.PriorityQueue) != 3 {
			t.Errorf("Expected queue length 3, got %d", len(cache.PriorityQueue))
		}

		if len(cache.HashMap) != 3 {
			t.Errorf("Expected hash map length 3, got %d", len(cache.HashMap))
		}

		if len(cache.OriginMap) != 3 {
			t.Errorf("Expected origin map length 3, got %d", len(cache.OriginMap))
		}

		// 验证key1已被删除
		if _, exists := cache.HashMap["key1"]; exists {
			t.Error("Expected key1 to be removed from cache")
		}
	})
}

func TestCacheImpl_Read(t *testing.T) {
	t.Run("测试Read方法", func(t *testing.T) {
		// 创建测试缓存
		keyFunc := func(s string) string { return s }
		cache := &CacheImpl[string, int]{
			Name:          "test-cache",
			Size:          3,
			PriorityQueue: []string{"key1", "key2"},
			HashMap: map[string]*ItemInfo[int]{
				"key1": {Item: 100, LastAccessTime: time.Now()},
				"key2": {Item: 200, LastAccessTime: time.Now()},
			},
			OriginMap: map[string]string{
				"key1": "key1",
				"key2": "key2",
			},
			KeyFunc: keyFunc,
			logger:  logr.Discard(),
		}

		// 读取存在的键
		value := cache.Read("key1")
		if value != 100 {
			t.Errorf("Expected value 100, got %d", value)
		}

		// 读取不存在的键
		value = cache.Read("key3")
		if value != 0 {
			t.Errorf("Expected default value 0, got %d", value)
		}

		// 验证访问后键的位置变化
		// key1应该被移动到队列末尾
		if len(cache.PriorityQueue) >= 2 && cache.PriorityQueue[len(cache.PriorityQueue)-1] != "key1" {
			t.Errorf("Expected key1 to be moved to end of queue")
		}
	})
}

func TestCacheImpl_Write(t *testing.T) {
	t.Run("测试Write方法", func(t *testing.T) {
		// 创建测试缓存
		keyFunc := func(s string) string { return s }
		cache := &CacheImpl[string, int]{
			Name:          "test-cache",
			Size:          3,
			PriorityQueue: make([]string, 0),
			HashMap:       make(map[string]*ItemInfo[int]),
			OriginMap:     make(map[string]string),
			KeyFunc:       keyFunc,
			logger:        logr.Discard(),
		}

		// 写入新项目
		cache.Write("key1", 100)

		if len(cache.PriorityQueue) != 1 {
			t.Errorf("Expected queue length 1, got %d", len(cache.PriorityQueue))
		}

		if cache.HashMap["key1"].Item != 100 {
			t.Errorf("Expected item value 100, got %d", cache.HashMap["key1"].Item)
		}

		// 更新现有项目
		cache.Write("key1", 150)

		if cache.HashMap["key1"].Item != 150 {
			t.Errorf("Expected updated item value 150, got %d", cache.HashMap["key1"].Item)
		}
	})
}

func TestCacheImpl_Forget(t *testing.T) {
	t.Run("测试Forget方法", func(t *testing.T) {
		// 创建测试缓存
		keyFunc := func(s string) string { return s }
		cache := &CacheImpl[string, int]{
			Name:          "test-cache",
			Size:          3,
			PriorityQueue: []string{"key1", "key2", "key3"},
			HashMap: map[string]*ItemInfo[int]{
				"key1": {Item: 100, LastAccessTime: time.Now()},
				"key2": {Item: 200, LastAccessTime: time.Now()},
				"key3": {Item: 300, LastAccessTime: time.Now()},
			},
			OriginMap: map[string]string{
				"key1": "key1",
				"key2": "key2",
				"key3": "key3",
			},
			KeyFunc: keyFunc,
			logger:  logr.Discard(),
		}

		// 忘记满足条件的项目 (值大于150)
		cache.Forget(func(s string, i int) bool {
			return i > 150
		})

		// 验证key3被删除
		if _, exists := cache.HashMap["key3"]; exists {
			t.Error("Expected key3 to be removed from cache")
		}

		if _, exists := cache.OriginMap["key3"]; exists {
			t.Error("Expected key3 to be removed from origin map")
		}

		// 验证key3从优先队列中移除
		found := false
		for _, key := range cache.PriorityQueue {
			if key == "key3" {
				found = true
				break
			}
		}
		if found {
			t.Error("Expected key3 to be removed from priority queue")
		}
	})
}

func TestCacheImpl_Process(t *testing.T) {
	t.Run("测试Process方法", func(t *testing.T) {
		// 创建测试缓存
		keyFunc := func(s string) string { return s }
		cache := &CacheImpl[string, int]{
			Name:          "test-cache",
			Size:          3,
			PriorityQueue: []string{"key1", "key2"},
			HashMap: map[string]*ItemInfo[int]{
				"key1": {Item: 100, LastAccessTime: time.Now()},
				"key2": {Item: 200, LastAccessTime: time.Now()},
			},
			OriginMap: map[string]string{
				"key1": "key1",
				"key2": "key2",
			},
			KeyFunc: keyFunc,
			logger:  logr.Discard(),
		}

		// 处理所有项目并计算总和
		sum := 0
		cache.Process(func(s string, i int) {
			sum += i
		})

		expectedSum := 300 // 100 + 200
		if sum != expectedSum {
			t.Errorf("Expected sum %d, got %d", expectedSum, sum)
		}
	})
}

func TestNewCache(t *testing.T) {
	t.Run("测试NewCache函数", func(t *testing.T) {
		// 重置全局插件缓存列表以避免并发问题
		oldCaches := cacheplugin.caches
		cacheplugin.caches = make([]CacheForPlugin, 0)
		defer func() {
			cacheplugin.caches = oldCaches
		}()

		// 创建一个处理函数
		podEvHandle := func(key NamespaceedNameNode, i int, logger logr.Logger) {
			// 空实现
		}

		// 创建新缓存
		keyFunc := func(s string) string { return s }
		cache := NewCache[string, int](context.Background(), "test-cache", 10, 1, keyFunc, podEvHandle)

		// 验证缓存已正确初始化
		if cache == nil {
			t.Error("Expected cache instance, got nil")
		}

		if cache.Name != "test-cache" {
			t.Errorf("Expected cache name 'test-cache', got %s", cache.Name)
		}

		if cache.Size != 10 {
			t.Errorf("Expected cache size 10, got %d", cache.Size)
		}

		// 验证缓存已被添加到全局插件中
		found := false
		cacheplugin.lock.Lock()
		for _, c := range cacheplugin.caches {
			if c == cache {
				found = true
				break
			}
		}
		cacheplugin.lock.Unlock()

		if !found {
			t.Error("Expected cache to be added to global plugin")
		}
	})
}
