/*
Copyright 2023 The Kubernetes Authors.

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

// Package clustertrustbundle abstracts access to ClusterTrustBundles so that
// projected volumes can use them.
package clustertrustbundle

import (
	"context"
	"encoding/pem"
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/go-logr/logr"
	certificatesv1alpha1 "k8s.io/api/certificates/v1alpha1"
	certificatesv1beta1 "k8s.io/api/certificates/v1beta1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	lrucache "k8s.io/apimachinery/pkg/util/cache"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

const (
	maxLabelSelectorLength = 100 * 1024
)

// clusterTrustBundle is a type constraint for version-independent ClusterTrustBundle API
type clusterTrustBundle interface {
	certificatesv1alpha1.ClusterTrustBundle | certificatesv1beta1.ClusterTrustBundle
}

// clusterTrustBundlesLister is an API-verion independent ClusterTrustBundles lister
type clusterTrustBundlesLister[T clusterTrustBundle] interface {
	Get(string) (*T, error)
	List(labels.Selector) ([]*T, error)
}

type clusterTrustBundleHandlers[T clusterTrustBundle] interface {
	GetName(*T) string
	GetSignerName(*T) string
	GetTrustBundle(*T) string
}

type alphaClusterTrustBundleHandlers struct{}

type betaClusterTrustBundleHandlers struct{}

func (b *alphaClusterTrustBundleHandlers) GetName(ctb *certificatesv1alpha1.ClusterTrustBundle) string {
	return ctb.Name
}

func (b *alphaClusterTrustBundleHandlers) GetSignerName(ctb *certificatesv1alpha1.ClusterTrustBundle) string {
	return ctb.Spec.SignerName
}

func (b *alphaClusterTrustBundleHandlers) GetTrustBundle(ctb *certificatesv1alpha1.ClusterTrustBundle) string {
	return ctb.Spec.TrustBundle
}

func (b betaClusterTrustBundleHandlers) GetName(ctb *certificatesv1beta1.ClusterTrustBundle) string {
	return ctb.Name
}

func (b *betaClusterTrustBundleHandlers) GetSignerName(ctb *certificatesv1beta1.ClusterTrustBundle) string {
	return ctb.Spec.SignerName
}

func (b *betaClusterTrustBundleHandlers) GetTrustBundle(ctb *certificatesv1beta1.ClusterTrustBundle) string {
	return ctb.Spec.TrustBundle
}

// Manager abstracts over the ability to get trust anchors.
type Manager interface {
	GetTrustAnchorsByName(name string, allowMissing bool) ([]byte, error)
	GetTrustAnchorsBySigner(signerName string, labelSelector *metav1.LabelSelector, allowMissing bool) ([]byte, error)
}

// InformerManager is the "real" manager.  It uses informers to track
// ClusterTrustBundle objects.
type InformerManager[T clusterTrustBundle] struct {
	ctbInformer cache.SharedIndexInformer
	ctbLister   clusterTrustBundlesLister[T]

	ctbHandlers clusterTrustBundleHandlers[T]

	normalizationCache *lrucache.LRUExpireCache
	cacheTTL           time.Duration
}

var _ Manager = (*InformerManager[certificatesv1beta1.ClusterTrustBundle])(nil)

func NewAlphaInformerManager(
	ctx context.Context, informerFactory informers.SharedInformerFactory, cacheSize int, cacheTTL time.Duration,
) (Manager, error) {
	bundlesInformer := informerFactory.Certificates().V1alpha1().ClusterTrustBundles()
	return newInformerManager(
		ctx, &alphaClusterTrustBundleHandlers{}, bundlesInformer.Informer(), bundlesInformer.Lister(), cacheSize, cacheTTL,
	)
}

func NewBetaInformerManager(
	ctx context.Context, informerFactory informers.SharedInformerFactory, cacheSize int, cacheTTL time.Duration,
) (Manager, error) {
	bundlesInformer := informerFactory.Certificates().V1beta1().ClusterTrustBundles()
	return newInformerManager(
		ctx, &betaClusterTrustBundleHandlers{}, bundlesInformer.Informer(), bundlesInformer.Lister(), cacheSize, cacheTTL,
	)
}

// newInformerManager returns an initialized InformerManager.
func newInformerManager[T clusterTrustBundle](ctx context.Context, handlers clusterTrustBundleHandlers[T], informer cache.SharedIndexInformer, lister clusterTrustBundlesLister[T], cacheSize int, cacheTTL time.Duration) (Manager, error) {
	// We need to call Informer() before calling start on the shared informer
	// factory, or the informer won't be registered to be started.
	m := &InformerManager[T]{
		ctbInformer:        informer,
		ctbLister:          lister,
		ctbHandlers:        handlers,
		normalizationCache: lrucache.NewLRUExpireCache(cacheSize),
		cacheTTL:           cacheTTL,
	}

	logger := klog.FromContext(ctx)
	// Have the informer bust cache entries when it sees updates that could
	// apply to them.
	_, err := m.ctbInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			ctb, ok := obj.(*T)
			if !ok {
				return
			}
			logger.Info("Dropping all cache entries for signer", "signerName", m.ctbHandlers.GetSignerName(ctb))
			m.dropCacheFor(ctb)
		},
		UpdateFunc: func(old, new any) {
			ctb, ok := new.(*T)
			if !ok {
				return
			}
			logger.Info("Dropping cache for ClusterTrustBundle", "signerName", m.ctbHandlers.GetSignerName(ctb))
			m.dropCacheFor(new.(*T))
		},
		DeleteFunc: func(obj any) {
			ctb, ok := obj.(*T)
			if !ok {
				tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
				if !ok {
					return
				}
				ctb, ok = tombstone.Obj.(*T)
				if !ok {
					return
				}
			}
			logger.Info("Dropping cache for ClusterTrustBundle", "signerName", m.ctbHandlers.GetSignerName(ctb))
			m.dropCacheFor(ctb)
		},
	})
	if err != nil {
		return nil, fmt.Errorf("while registering event handler on informer: %w", err)
	}

	return m, nil
}

func (m *InformerManager[T]) dropCacheFor(ctb *T) {
	if ctbSignerName := m.ctbHandlers.GetSignerName(ctb); ctbSignerName != "" {
		m.normalizationCache.RemoveAll(func(key any) bool {
			return key.(cacheKeyType).signerName == ctbSignerName
		})
	} else {
		m.normalizationCache.RemoveAll(func(key any) bool {
			return key.(cacheKeyType).ctbName == m.ctbHandlers.GetName(ctb)
		})
	}
}

// GetTrustAnchorsByName returns normalized and deduplicated trust anchors from
// a single named ClusterTrustBundle.
func (m *InformerManager[T]) GetTrustAnchorsByName(name string, allowMissing bool) ([]byte, error) {
	if !m.ctbInformer.HasSynced() {
		return nil, fmt.Errorf("ClusterTrustBundle informer has not yet synced")
	}

	cacheKey := cacheKeyType{ctbName: name}

	if cachedAnchors, ok := m.normalizationCache.Get(cacheKey); ok {
		return cachedAnchors.([]byte), nil
	}

	ctb, err := m.ctbLister.Get(name)
	if k8serrors.IsNotFound(err) && allowMissing {
		return []byte{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("while getting ClusterTrustBundle: %w", err)
	}

	pemTrustAnchors, err := m.normalizeTrustAnchors([]*T{ctb})
	if err != nil {
		return nil, fmt.Errorf("while normalizing trust anchors: %w", err)
	}

	m.normalizationCache.Add(cacheKey, pemTrustAnchors, m.cacheTTL)

	return pemTrustAnchors, nil
}

// GetTrustAnchorsBySigner returns normalized and deduplicated trust anchors
// from a set of selected ClusterTrustBundles.
func (m *InformerManager[T]) GetTrustAnchorsBySigner(signerName string, labelSelector *metav1.LabelSelector, allowMissing bool) ([]byte, error) {
	if !m.ctbInformer.HasSynced() {
		return nil, fmt.Errorf("ClusterTrustBundle informer has not yet synced")
	}

	// Note that this function treats nil as "match nothing", and non-nil but
	// empty as "match everything".
	selector, err := metav1.LabelSelectorAsSelector(labelSelector)
	if err != nil {
		return nil, fmt.Errorf("while parsing label selector: %w", err)
	}

	cacheKey := cacheKeyType{signerName: signerName, labelSelector: selector.String()}

	if lsLen := len(cacheKey.labelSelector); lsLen > maxLabelSelectorLength {
		return nil, fmt.Errorf("label selector length (%d) is larger than %d", lsLen, maxLabelSelectorLength)
	}

	if cachedAnchors, ok := m.normalizationCache.Get(cacheKey); ok {
		return cachedAnchors.([]byte), nil
	}

	rawCTBList, err := m.ctbLister.List(selector)
	if err != nil {
		return nil, fmt.Errorf("while listing ClusterTrustBundles matching label selector %v: %w", labelSelector, err)
	}

	ctbList := []*T{}
	for _, ctb := range rawCTBList {
		if m.ctbHandlers.GetSignerName(ctb) == signerName {
			ctbList = append(ctbList, ctb)
		}
	}

	if len(ctbList) == 0 {
		if allowMissing {
			return []byte{}, nil
		}
		return nil, fmt.Errorf("combination of signerName and labelSelector matched zero ClusterTrustBundles")
	}

	pemTrustAnchors, err := m.normalizeTrustAnchors(ctbList)
	if err != nil {
		return nil, fmt.Errorf("while normalizing trust anchors: %w", err)
	}

	m.normalizationCache.Add(cacheKey, pemTrustAnchors, m.cacheTTL)

	return pemTrustAnchors, nil
}

func (m *InformerManager[T]) normalizeTrustAnchors(ctbList []*T) ([]byte, error) {
	// Deduplicate trust anchors from all ClusterTrustBundles.
	trustAnchorSet := sets.Set[string]{}
	for _, ctb := range ctbList {
		rest := []byte(m.ctbHandlers.GetTrustBundle(ctb))
		var b *pem.Block
		for {
			b, rest = pem.Decode(rest)
			if b == nil {
				break
			}
			trustAnchorSet = trustAnchorSet.Insert(string(b.Bytes))
		}
	}

	// Give the list a stable ordering that changes each time Kubelet restarts.
	trustAnchorList := sets.List(trustAnchorSet)
	rand.Shuffle(len(trustAnchorList), func(i, j int) {
		trustAnchorList[i], trustAnchorList[j] = trustAnchorList[j], trustAnchorList[i]
	})

	pemTrustAnchors := []byte{}
	for _, ta := range trustAnchorList {
		b := &pem.Block{
			Type:  "CERTIFICATE",
			Bytes: []byte(ta),
		}
		pemTrustAnchors = append(pemTrustAnchors, pem.EncodeToMemory(b)...)
	}

	return pemTrustAnchors, nil
}

type cacheKeyType struct {
	ctbName       string
	signerName    string
	labelSelector string
}

// NoopManager always returns an error, for use in static kubelet mode.
type NoopManager struct{}

var _ Manager = (*NoopManager)(nil)

// GetTrustAnchorsByName implements Manager.
func (m *NoopManager) GetTrustAnchorsByName(name string, allowMissing bool) ([]byte, error) {
	return nil, fmt.Errorf("ClusterTrustBundle projection is not supported in static kubelet mode")
}

// GetTrustAnchorsBySigner implements Manager.
func (m *NoopManager) GetTrustAnchorsBySigner(signerName string, labelSelector *metav1.LabelSelector, allowMissing bool) ([]byte, error) {
	return nil, fmt.Errorf("ClusterTrustBundle projection is not supported in static kubelet mode")
}

// LazyInformerManager decides whether to use the noop or the actual manager on a call to
// the manager's methods.
// We cannot determine this upon startup because some may rely on the kubelet to be fully
// running in order to setup their kube-apiserver.
type LazyInformerManager struct {
	manager           Manager
	managerLock       sync.RWMutex
	client            clientset.Interface
	cacheSize         int
	contextWithLogger context.Context
	logger            logr.Logger
}

func NewLazyInformerManager(ctx context.Context, kubeClient clientset.Interface, cacheSize int) Manager {
	return &LazyInformerManager{
		client:            kubeClient,
		cacheSize:         cacheSize,
		contextWithLogger: ctx,
		logger:            klog.FromContext(ctx),
		managerLock:       sync.RWMutex{},
	}
}

func (m *LazyInformerManager) GetTrustAnchorsByName(name string, allowMissing bool) ([]byte, error) {
	if err := m.ensureManagerSet(); err != nil {
		return nil, fmt.Errorf("failed to ensure informer manager for ClusterTrustBundles: %w", err)
	}
	return m.manager.GetTrustAnchorsByName(name, allowMissing)
}

func (m *LazyInformerManager) GetTrustAnchorsBySigner(signerName string, labelSelector *metav1.LabelSelector, allowMissing bool) ([]byte, error) {
	if err := m.ensureManagerSet(); err != nil {
		return nil, fmt.Errorf("failed to ensure informer manager for ClusterTrustBundles: %w", err)
	}
	return m.manager.GetTrustAnchorsBySigner(signerName, labelSelector, allowMissing)
}

func (m *LazyInformerManager) isManagerSet() bool {
	m.managerLock.RLock()
	defer m.managerLock.RUnlock()
	return m.manager != nil
}

type managerConstructor func(ctx context.Context, informerFactory informers.SharedInformerFactory, cacheSize int, cacheTTL time.Duration) (Manager, error)

func (m *LazyInformerManager) ensureManagerSet() error {
	if m.isManagerSet() {
		return nil
	}

	m.managerLock.Lock()
	defer m.managerLock.Unlock()
	// we need to check again in case the manager was set between locking
	if m.manager != nil {
		return nil
	}

	managerSchema := map[schema.GroupVersion]managerConstructor{
		certificatesv1alpha1.SchemeGroupVersion: NewAlphaInformerManager,
		certificatesv1beta1.SchemeGroupVersion:  NewBetaInformerManager,
	}

	kubeInformers := informers.NewSharedInformerFactoryWithOptions(m.client, 0)

	var clusterTrustBundleManager Manager
	var foundGV string
	for _, gv := range []schema.GroupVersion{certificatesv1beta1.SchemeGroupVersion, certificatesv1alpha1.SchemeGroupVersion} {
		ctbAPIAvailable, err := clusterTrustBundlesAvailable(m.client, gv)
		if err != nil {
			return fmt.Errorf("failed to determine which informer manager to choose: %w", err)
		}

		if !ctbAPIAvailable {
			continue
		}

		clusterTrustBundleManager, err = managerSchema[gv](m.contextWithLogger, kubeInformers, m.cacheSize, 5*time.Minute)
		if err != nil {
			return fmt.Errorf("error starting informer-based ClusterTrustBundle manager: %w", err)
		}
		foundGV = gv.String()
		break
	}

	if clusterTrustBundleManager == nil {
		m.manager = &NoopManager{}
		m.logger.Info("No version of the ClusterTrustBundle API was found, the ClusterTrustBundle informer won't be started")
		return nil
	}

	m.manager = clusterTrustBundleManager
	kubeInformers.Start(m.contextWithLogger.Done())
	m.logger.Info("Started ClusterTrustBundle informer", "apiGroup", foundGV)

	// a cache fetch will likely follow right after, wait for the freshly started
	// informers to sync
	synced := true
	timeoutContext, cancel := context.WithTimeout(m.contextWithLogger, 10*time.Second)
	defer cancel()
	m.logger.Info("Waiting for ClusterTrustBundle informer to sync")
	for _, ok := range kubeInformers.WaitForCacheSync(timeoutContext.Done()) {
		synced = synced && ok
	}
	if synced {
		m.logger.Info("ClusterTrustBundle informer synced")
	} else {
		m.logger.Info("ClusterTrustBundle informer not synced, continuing to attempt in background")
	}

	return nil
}

func clusterTrustBundlesAvailable(client clientset.Interface, gv schema.GroupVersion) (bool, error) {
	resList, err := client.Discovery().ServerResourcesForGroupVersion(gv.String())
	if k8serrors.IsNotFound(err) {
		return false, nil
	}

	if resList != nil {
		// even in case of an error above there might be a partial list for APIs that
		// were already successfully discovered
		for _, r := range resList.APIResources {
			if r.Name == "clustertrustbundles" {
				return true, nil
			}
		}
	}
	return false, err
}
