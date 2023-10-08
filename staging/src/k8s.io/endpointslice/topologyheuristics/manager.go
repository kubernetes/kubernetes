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

package topologyheuristics

import (
	"fmt"
	"slices"
	"sync"

	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	"k8s.io/klog/v2"
)

// Manager manages multiple EndpointSlice Topology Heuristics. It exposes
// functions for adding and removing topology hints to endpoint slices using
// various heuristics.
type Manager struct {
	// heuristicsByName maintains a mapping of a heuristic name to a Heuristic
	// instance.
	heuristicsByName map[string]Heuristic
	// defaultHeuristicName is the name of the Heuristic which will be considered
	// the default.
	defaultHeuristicName string
	// nonEPSHintHeuristicNames should contain the names of heuristics which are not
	// implemented through EndpointSlice hints and may be implemented in the
	// Dataplane (or in other ways). Manager will NOT send "unrecognizable" events
	// for these heuristic names (despite the fact that it does not populate
	// EndpointSlice hints for these.)
	nonEPSHintHeuristicNames []string

	mu sync.Mutex
	// activeHeuristicsByService keeps track of the heuristic currently in use by
	// a Service.
	activeHeuristicsByService map[string]map[discoveryv1.AddressType]Heuristic
}

// NewManager returns an instance of a heuristic Manager.
//
// `heuristics` contains the set of heuristics that this Mangaer will use and
// manage. They should have unique names (as returned by their Heuristic.Name()
// function). In the case when two heuristics have the same name, the first
// heuristic will be used.
func NewManager(logger klog.Logger, heuristics []Heuristic, defaultHeuristicName string, nonEPSHintHeuristicNames []string) (*Manager, error) {
	heuristicsByName := make(map[string]Heuristic)
	var defaultHeuristicExists bool
	for _, heuristic := range heuristics {
		// Check if a heuristic with the same name already exists.
		_, ok := heuristicsByName[heuristic.Name()]
		if ok {
			return nil, fmt.Errorf("found multiple heuristics with the same name %q", heuristic.Name())
		}
		// Check for occurrence of default heuristic.
		if heuristic.Name() == defaultHeuristicName {
			defaultHeuristicExists = true
		}

		heuristicsByName[heuristic.Name()] = heuristic
	}

	if !defaultHeuristicExists {
		return nil, fmt.Errorf("heuristic with name %q does not exist", defaultHeuristicName)
	}

	return &Manager{
		heuristicsByName:          heuristicsByName,
		defaultHeuristicName:      defaultHeuristicName,
		nonEPSHintHeuristicNames:  nonEPSHintHeuristicNames,
		activeHeuristicsByService: make(map[string]map[discoveryv1.AddressType]Heuristic),
	}, nil
}

// PopulateHints populates topology hints on EndpointSlices and returns updated
// lists of EndpointSlices to create and update.
//
// It also updates all heuristic caches to reflect the latest heuristic in use.
// This means that if a service was previously using heuristic A and is now
// using heuristic B, the cache for heuristic A will be updated to remove any
// cached hints for that service.
func (m *Manager) PopulateHints(logger klog.Logger, service *corev1.Service, si *SliceInfo) ([]*discoveryv1.EndpointSlice, []*discoveryv1.EndpointSlice, []*EventBuilder) {
	activeHeuristic, activeHeuristicExists := m.activeHeuristicForService(si.ServiceKey, si.AddressType)
	desiredHeuristic, desiredHeuristicEnabled, desiredHeuristicExists := m.desiredHeuristicForService(service)
	var events []*EventBuilder

	switch {
	// Transition: (no heuristic) -> (no heuristic)
	//
	// Remove any existing hints on the endpoint slices.
	//
	// Note that ideally, this should be a no-op since the slice didn't have any
	// hints populated previously. But to ensure the desired state is accurate,
	// we remove any existing hints.
	case !activeHeuristicExists && !desiredHeuristicExists:
		slicesToCreate, slicesToUpdate := RemoveHintsFromSlices(si)
		return slicesToCreate, slicesToUpdate, events

	// Transition: (no heuristic) -> (some heuristic)
	//
	// Set topology hints using the new heuristic.
	case !activeHeuristicExists && desiredHeuristicExists:
		m.setActiveHeuristicForService(si.ServiceKey, si.AddressType, desiredHeuristic)
		return desiredHeuristic.PopulateHints(logger, si)

	// Transition: (some heuristic) -> (no heuristic)
	//
	// Clear cached hints from the old heuristic and remove any existing hints in
	// the endpoint slices.
	case activeHeuristicExists && !desiredHeuristicExists:
		desiredHeuristicName, _ := HeuristicNameFromAnnotations(service.Annotations)
		if desiredHeuristicEnabled {
			// This is the case when the topology annotation has a non-disabled value
			// but the heuristic does not exist in the heuristic manager. Check if
			// this is a non-endpointslice heuristic.
			if !slices.Contains(m.nonEPSHintHeuristicNames, desiredHeuristicName) {
				// Send event to notify about an unrecognizable heuristic.
				logger.Info(TopologyHeuristicChangedToNotSupported+", removing existing hints", "key", si.ServiceKey, "addressType", si.AddressType, "oldHeuristic", activeHeuristic.Name(), "newHeuristic", desiredHeuristicName)
				events = append(events, &EventBuilder{
					EventType: corev1.EventTypeWarning,
					Reason:    "TopologyAwareHintsDisabled",
					Message:   fmt.Sprintf("%v; addressType: %v, oldHeuristic: %v, newHeuristic: %v", TopologyHeuristicChangedToNotSupported, si.AddressType, activeHeuristic.Name(), desiredHeuristicName),
				})
			}
		} else {
			// This is the case when the topology annotation on the service determines
			// a disabled value.
			logger.Info(TopologyAwareHintsDisabled+", removing existing hints", "key", si.ServiceKey, "addressType", si.AddressType, "heuristic", desiredHeuristicName)
			events = append(events, &EventBuilder{
				EventType: corev1.EventTypeWarning,
				Reason:    "TopologyAwareHintsDisabled",
				Message:   FormatWithAddressTypeAndHeuristicName(TopologyAwareHintsDisabled, si.AddressType, activeHeuristic.Name()),
			})
		}

		activeHeuristic.ClearCachedHints(logger, si.ServiceKey, si.AddressType)
		m.setActiveHeuristicForService(si.ServiceKey, si.AddressType, nil)
		slicesToCreate, slicesToUpdate := RemoveHintsFromSlices(si)
		return slicesToCreate, slicesToUpdate, events

	// Transition: (some heuristic) -> (some heuristic)
	//
	// Clear cached hints from the old heuristic if there was a change in
	// heuristic. Then, set topology hints using the new heuristic.
	case activeHeuristicExists && desiredHeuristicExists:
		desiredHeuristicName, _ := HeuristicNameFromAnnotations(service.Annotations)
		logger.Info(TopologyHeuristicChanged, "key", si.ServiceKey, "addressType", si.AddressType, "oldHeuristic", activeHeuristic.Name(), "newHeuristic", desiredHeuristicName)
		events = append(events, &EventBuilder{
			EventType: corev1.EventTypeNormal,
			Reason:    "TopologyHeuristicChanged",
			Message:   fmt.Sprintf("%v; addressType: %v, oldHeuristic: %v, newHeuristic: %v", TopologyHeuristicChanged, si.AddressType, activeHeuristic.Name(), desiredHeuristicName),
		})

		if activeHeuristic.Name() != desiredHeuristic.Name() {
			activeHeuristic.ClearCachedHints(logger, si.ServiceKey, si.AddressType)
		}
		m.setActiveHeuristicForService(si.ServiceKey, si.AddressType, desiredHeuristic)
		slicesToCreate, slicesToUpdate, topologyEvents := desiredHeuristic.PopulateHints(logger, si)
		events = append(events, topologyEvents...)
		return slicesToCreate, slicesToUpdate, events
	}

	// Ideally, the following return will never get executed.
	return si.ToCreate, si.ToUpdate, events
}

// ClearCachedHints removes any cached topology hints from the "active"
// Heuristic of the service.
//
// "active" Heuristic referes to the heuristic which is currently in effect for
// this service and addrType.
func (m *Manager) ClearCachedHints(logger klog.Logger, serviceKey string, addrType discoveryv1.AddressType) []*EventBuilder {
	activeHeuristic, activeHeuristicExists := m.activeHeuristicForService(serviceKey, addrType)
	if !activeHeuristicExists {
		return nil
	}

	events := []*EventBuilder{{
		EventType: corev1.EventTypeWarning,
		Reason:    "TopologyAwareHintsDisabled",
		Message:   FormatWithAddressTypeAndHeuristicName(TopologyAwareHintsDisabled, addrType, activeHeuristic.Name()),
	}}

	m.setActiveHeuristicForService(serviceKey, addrType, nil)
	activeHeuristic.ClearCachedHints(logger, serviceKey, addrType)
	return events
}

// activeHeuristicForService returns the Heuristic which is presently being used
// by the service.
//
// The Heuristic returned represents the "current state", which could be
// different from the "desired state". Use desiredHeuristicForService() to get
// the "desired state".
func (m *Manager) activeHeuristicForService(serviceKey string, addrType discoveryv1.AddressType) (Heuristic, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.activeHeuristicsByService[serviceKey]; !ok {
		m.activeHeuristicsByService[serviceKey] = make(map[discoveryv1.AddressType]Heuristic)
	}

	heuristic, ok := m.activeHeuristicsByService[serviceKey][addrType]
	return heuristic, ok && heuristic != nil
}

// setActiveHeuristicForService sets the active Heuristic for the service and
// addrType.
//
// Refer comment under activeHeuristicForService() for an understanding of what
// active Heuristic means.
func (m *Manager) setActiveHeuristicForService(serviceKey string, addrType discoveryv1.AddressType, heuristic Heuristic) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.activeHeuristicsByService[serviceKey]; !ok {
		m.activeHeuristicsByService[serviceKey] = make(map[discoveryv1.AddressType]Heuristic)
	}

	m.activeHeuristicsByService[serviceKey][addrType] = heuristic
}

// desiredHeuristicForService returns the Heuristic corresponding to the
// topology annotation in the service.
//   - enabled return value signifies that the topology annotation determines an
//     enabled value.
//   - exists return value signifies existence of the heuristic in the heuristic
//     manager.
//
// The Heuristic returned represents the "desired state", which could be
// different from the "current state". Use activeHeuristicForService() to get
// the "current state".
func (m *Manager) desiredHeuristicForService(service *corev1.Service) (topology Heuristic, enabled, exists bool) {
	desiredHeuristicName, desiredHeuristicEnabled := HeuristicNameFromAnnotations(service.GetAnnotations())
	if !desiredHeuristicEnabled {
		return nil, false, false
	}

	// Special case for backward compatibility: If the topology annotation value
	// is set to "Auto", we need to return defaultHeuristic.
	if desiredHeuristicName == "Auto" || desiredHeuristicName == "auto" {
		heuristic, ok := m.heuristicsByName[m.defaultHeuristicName]
		return heuristic, true, ok
	}

	for heuristicName, heuristic := range m.heuristicsByName {
		if desiredHeuristicName == heuristicName {
			return heuristic, true, true
		}
	}

	return nil, true, false
}

// RecognizableHeuristic returns true if the given heuristicName is recognized
// by the Manager. A topology is recognized if it is:
//   - Supported by the Manager.
//   - (or)
//   - Present in nonEPSHintHeuristicNames (i.e. not implemented through
//     EndpointSlice hints)
func (m *Manager) RecognizableHeuristic(heuristicName string) bool {
	var recognizableHeuristicNames []string
	recognizableHeuristicNames = append(recognizableHeuristicNames, "Auto", "auto")
	recognizableHeuristicNames = append(recognizableHeuristicNames, m.nonEPSHintHeuristicNames...)
	for name := range m.heuristicsByName {
		recognizableHeuristicNames = append(recognizableHeuristicNames, name)
	}
	return slices.Contains(recognizableHeuristicNames, heuristicName)
}

// HeuristicNameFromAnnotations function returns the value of the topology
// annotation, if one is present. It also identifies whether the annotation
// value indicates that the topology aware routing is enabled or disabled.
//
// The function uses two annotation keys to determine the topology annotation
// value:
//   - v1.AnnotationTopologyMode
//   - v1.DeprecatedAnnotationTopologyAwareHints.
//
// If both annotation keys are set, v1.DeprecatedAnnotationTopologyAwareHints
// has precedence.
func HeuristicNameFromAnnotations(annotations map[string]string) (heuristicName string, enabled bool) {
	annotationKey := TopologyAnnotationKeyFromAnnnotations(annotations)

	switch annotationKey {

	case corev1.DeprecatedAnnotationTopologyAwareHints:
		heuristicName := annotations[corev1.DeprecatedAnnotationTopologyAwareHints]
		return heuristicName, heuristicName == "Auto" || heuristicName == "auto"

	case corev1.AnnotationTopologyMode:
		heuristicName := annotations[corev1.AnnotationTopologyMode]
		return heuristicName, heuristicName != "" && heuristicName != "disabled" && heuristicName != "Disabled"

	}

	// This should ideally not get executed.
	return "", false
}

// TopologyAnnotationKeyFromAnnnotations returns the annotation key which
// decides the name of the topology heuristic to be used.
//
// The function looks for the presence of the following two keys:
//   - v1.AnnotationTopologyMode
//   - v1.DeprecatedAnnotationTopologyAwareHints.
//
// If both annotation keys are present, v1.DeprecatedAnnotationTopologyAwareHints
// has precedence.
func TopologyAnnotationKeyFromAnnnotations(annotations map[string]string) string {
	_, ok := annotations[corev1.DeprecatedAnnotationTopologyAwareHints]
	if ok {
		return corev1.DeprecatedAnnotationTopologyAwareHints
	}
	return corev1.AnnotationTopologyMode
}

type Heuristic interface {
	// Name returns the name associated with the Heuristic. This will be matched
	// against the topology annotation in the Service to decide if the Heuristic
	// is currently active for a service.
	Name() string

	// PopulateHints populates topology hints on EndpointSlices and returns
	// updated lists of EndpointSlices to create and update. It also returns any
	// Events that need to be recorded for the Service.
	//
	// Combination of [SliceInfo.ServiceKey and SliceInfo.AddressType] are used to
	// identify the endpoint slices which these hints will be associated with.
	PopulateHints(logger klog.Logger, si *SliceInfo) (slicesToCreate []*discoveryv1.EndpointSlice, slicesToUpdate []*discoveryv1.EndpointSlice, events []*EventBuilder)

	// ClearCachedHints removes any cached topology hints associated with the
	// [service and addrType].
	//
	// This may only be useful for heuristics which need to maintain an internal
	// state to track the active topology hints. This function may be a no-op for
	// heuristics which do not have a need to maintain such a state.
	ClearCachedHints(logger klog.Logger, serviceKey string, addrType discoveryv1.AddressType)
}
