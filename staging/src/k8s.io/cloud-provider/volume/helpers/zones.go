/*
Copyright 2019 The Kubernetes Authors.

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

package helpers

import (
	"fmt"
	"hash/fnv"
	"math/rand"
	"strconv"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	cloudvolume "k8s.io/cloud-provider/volume"
	"k8s.io/klog/v2"
)

// LabelZonesToSet converts a PV label value from string containing a delimited list of zones to set
func LabelZonesToSet(labelZonesValue string) (sets.String, error) {
	return stringToSet(labelZonesValue, cloudvolume.LabelMultiZoneDelimiter)
}

// ZonesSetToLabelValue converts zones set to label value
func ZonesSetToLabelValue(strSet sets.String) string {
	return strings.Join(strSet.UnsortedList(), cloudvolume.LabelMultiZoneDelimiter)
}

// ZonesToSet converts a string containing a comma separated list of zones to set
func ZonesToSet(zonesString string) (sets.String, error) {
	zones, err := stringToSet(zonesString, ",")
	if err != nil {
		return nil, fmt.Errorf("error parsing zones %s, must be strings separated by commas: %v", zonesString, err)
	}
	return zones, nil
}

// StringToSet converts a string containing list separated by specified delimiter to a set
func stringToSet(str, delimiter string) (sets.String, error) {
	zonesSlice := strings.Split(str, delimiter)
	zonesSet := make(sets.String)
	for _, zone := range zonesSlice {
		trimmedZone := strings.TrimSpace(zone)
		if trimmedZone == "" {
			return make(sets.String), fmt.Errorf(
				"%q separated list (%q) must not contain an empty string",
				delimiter,
				str)
		}
		zonesSet.Insert(trimmedZone)
	}
	return zonesSet, nil
}

// LabelZonesToList converts a PV label value from string containing a delimited list of zones to list
func LabelZonesToList(labelZonesValue string) ([]string, error) {
	return stringToList(labelZonesValue, cloudvolume.LabelMultiZoneDelimiter)
}

// StringToList converts a string containing list separated by specified delimiter to a list
func stringToList(str, delimiter string) ([]string, error) {
	zonesSlice := make([]string, 0)
	for _, zone := range strings.Split(str, delimiter) {
		trimmedZone := strings.TrimSpace(zone)
		if trimmedZone == "" {
			return nil, fmt.Errorf(
				"%q separated list (%q) must not contain an empty string",
				delimiter,
				str)
		}
		zonesSlice = append(zonesSlice, trimmedZone)
	}
	return zonesSlice, nil
}

// SelectZoneForVolume is a wrapper around SelectZonesForVolume
// to select a single zone for a volume based on parameters
func SelectZoneForVolume(zoneParameterPresent, zonesParameterPresent bool, zoneParameter string, zonesParameter, zonesWithNodes sets.String, node *v1.Node, allowedTopologies []v1.TopologySelectorTerm, pvcName string) (string, error) {
	zones, err := SelectZonesForVolume(zoneParameterPresent, zonesParameterPresent, zoneParameter, zonesParameter, zonesWithNodes, node, allowedTopologies, pvcName, 1)
	if err != nil {
		return "", err
	}
	zone, ok := zones.PopAny()
	if !ok {
		return "", fmt.Errorf("could not determine a zone to provision volume in")
	}
	return zone, nil
}

// SelectZonesForVolume selects zones for a volume based on several factors:
// node.zone, allowedTopologies, zone/zones parameters from storageclass,
// zones with active nodes from the cluster. The number of zones = replicas.
func SelectZonesForVolume(zoneParameterPresent, zonesParameterPresent bool, zoneParameter string, zonesParameter, zonesWithNodes sets.String, node *v1.Node, allowedTopologies []v1.TopologySelectorTerm, pvcName string, numReplicas uint32) (sets.String, error) {
	if zoneParameterPresent && zonesParameterPresent {
		return nil, fmt.Errorf("both zone and zones StorageClass parameters must not be used at the same time")
	}

	var zoneFromNode string
	// pick one zone from node if present
	if node != nil {
		// VolumeScheduling implicit since node is not nil
		if zoneParameterPresent || zonesParameterPresent {
			return nil, fmt.Errorf("zone[s] cannot be specified in StorageClass if VolumeBindingMode is set to WaitForFirstConsumer. Please specify allowedTopologies in StorageClass for constraining zones")
		}

		// pick node's zone for one of the replicas
		var ok bool
		zoneFromNode, ok = node.ObjectMeta.Labels[v1.LabelTopologyZone]
		if !ok {
			zoneFromNode, ok = node.ObjectMeta.Labels[v1.LabelFailureDomainBetaZone]
			if !ok {
				return nil, fmt.Errorf("Either %s or %s Label for node missing", v1.LabelTopologyZone, v1.LabelFailureDomainBetaZone)
			}
		}
		// if single replica volume and node with zone found, return immediately
		if numReplicas == 1 {
			return sets.NewString(zoneFromNode), nil
		}
	}

	// pick zone from allowedZones if specified
	allowedZones, err := ZonesFromAllowedTopologies(allowedTopologies)
	if err != nil {
		return nil, err
	}

	if (len(allowedTopologies) > 0) && (allowedZones.Len() == 0) {
		return nil, fmt.Errorf("no matchLabelExpressions with %s key found in allowedTopologies. Please specify matchLabelExpressions with %s key", v1.LabelTopologyZone, v1.LabelTopologyZone)
	}

	if allowedZones.Len() > 0 {
		// VolumeScheduling implicit since allowedZones present
		if zoneParameterPresent || zonesParameterPresent {
			return nil, fmt.Errorf("zone[s] cannot be specified in StorageClass if allowedTopologies specified")
		}
		// scheduler will guarantee if node != null above, zoneFromNode is member of allowedZones.
		// so if zoneFromNode != "", we can safely assume it is part of allowedZones.
		zones, err := chooseZonesForVolumeIncludingZone(allowedZones, pvcName, zoneFromNode, numReplicas)
		if err != nil {
			return nil, fmt.Errorf("cannot process zones in allowedTopologies: %v", err)
		}
		return zones, nil
	}

	// pick zone from parameters if present
	if zoneParameterPresent {
		if numReplicas > 1 {
			return nil, fmt.Errorf("zone cannot be specified if desired number of replicas for pv is greather than 1. Please specify zones or allowedTopologies to specify desired zones")
		}
		return sets.NewString(zoneParameter), nil
	}

	if zonesParameterPresent {
		if uint32(zonesParameter.Len()) < numReplicas {
			return nil, fmt.Errorf("not enough zones found in zones parameter to provision a volume with %d replicas. Found %d zones, need %d zones", numReplicas, zonesParameter.Len(), numReplicas)
		}
		// directly choose from zones parameter; no zone from node need to be considered
		return ChooseZonesForVolume(zonesParameter, pvcName, numReplicas), nil
	}

	// pick zone from zones with nodes
	if zonesWithNodes.Len() > 0 {
		// If node != null (and thus zoneFromNode != ""), zoneFromNode will be member of zonesWithNodes
		zones, err := chooseZonesForVolumeIncludingZone(zonesWithNodes, pvcName, zoneFromNode, numReplicas)
		if err != nil {
			return nil, fmt.Errorf("cannot process zones where nodes exist in the cluster: %v", err)
		}
		return zones, nil
	}
	return nil, fmt.Errorf("cannot determine zones to provision volume in")
}

// ZonesFromAllowedTopologies returns a list of zones specified in allowedTopologies
func ZonesFromAllowedTopologies(allowedTopologies []v1.TopologySelectorTerm) (sets.String, error) {
	zones := make(sets.String)
	for _, term := range allowedTopologies {
		for _, exp := range term.MatchLabelExpressions {
			if exp.Key == v1.LabelTopologyZone || exp.Key == v1.LabelFailureDomainBetaZone {
				for _, value := range exp.Values {
					zones.Insert(value)
				}
			} else {
				return nil, fmt.Errorf("unsupported key found in matchLabelExpressions: %s", exp.Key)
			}
		}
	}
	return zones, nil
}

// chooseZonesForVolumeIncludingZone is a wrapper around ChooseZonesForVolume that ensures zoneToInclude is chosen
// zoneToInclude can either be empty in which case it is ignored. If non-empty, zoneToInclude is expected to be member of zones.
// numReplicas is expected to be > 0 and <= zones.Len()
func chooseZonesForVolumeIncludingZone(zones sets.String, pvcName, zoneToInclude string, numReplicas uint32) (sets.String, error) {
	if numReplicas == 0 {
		return nil, fmt.Errorf("invalid number of replicas passed")
	}
	if uint32(zones.Len()) < numReplicas {
		return nil, fmt.Errorf("not enough zones found to provision a volume with %d replicas. Need at least %d distinct zones for a volume with %d replicas", numReplicas, numReplicas, numReplicas)
	}
	if zoneToInclude != "" && !zones.Has(zoneToInclude) {
		return nil, fmt.Errorf("zone to be included: %s needs to be member of set: %v", zoneToInclude, zones)
	}
	if uint32(zones.Len()) == numReplicas {
		return zones, nil
	}
	if zoneToInclude != "" {
		zones.Delete(zoneToInclude)
		numReplicas = numReplicas - 1
	}
	zonesChosen := ChooseZonesForVolume(zones, pvcName, numReplicas)
	if zoneToInclude != "" {
		zonesChosen.Insert(zoneToInclude)
	}
	return zonesChosen, nil
}

// ChooseZonesForVolume is identical to ChooseZoneForVolume, but selects a multiple zones, for multi-zone disks.
func ChooseZonesForVolume(zones sets.String, pvcName string, numZones uint32) sets.String {
	// No zones available, return empty set.
	replicaZones := sets.NewString()
	if zones.Len() == 0 {
		return replicaZones
	}

	// We create the volume in a zone determined by the name
	// Eventually the scheduler will coordinate placement into an available zone
	hash, index := getPVCNameHashAndIndexOffset(pvcName)

	// Zones.List returns zones in a consistent order (sorted)
	// We do have a potential failure case where volumes will not be properly spread,
	// if the set of zones changes during StatefulSet volume creation.  However, this is
	// probably relatively unlikely because we expect the set of zones to be essentially
	// static for clusters.
	// Hopefully we can address this problem if/when we do full scheduler integration of
	// PVC placement (which could also e.g. avoid putting volumes in overloaded or
	// unhealthy zones)
	zoneSlice := zones.List()

	startingIndex := index * numZones
	for index = startingIndex; index < startingIndex+numZones; index++ {
		zone := zoneSlice[(hash+index)%uint32(len(zoneSlice))]
		replicaZones.Insert(zone)
	}

	klog.V(2).Infof("Creating volume for replicated PVC %q; chosen zones=%q from zones=%q",
		pvcName, replicaZones.UnsortedList(), zoneSlice)
	return replicaZones
}

func getPVCNameHashAndIndexOffset(pvcName string) (hash uint32, index uint32) {
	if pvcName == "" {
		// We should always be called with a name; this shouldn't happen
		klog.Warningf("No name defined during volume create; choosing random zone")

		hash = rand.Uint32()
	} else {
		hashString := pvcName

		// Heuristic to make sure that volumes in a StatefulSet are spread across zones
		// StatefulSet PVCs are (currently) named ClaimName-StatefulSetName-Id,
		// where Id is an integer index.
		// Note though that if a StatefulSet pod has multiple claims, we need them to be
		// in the same zone, because otherwise the pod will be unable to mount both volumes,
		// and will be unschedulable.  So we hash _only_ the "StatefulSetName" portion when
		// it looks like `ClaimName-StatefulSetName-Id`.
		// We continue to round-robin volume names that look like `Name-Id` also; this is a useful
		// feature for users that are creating statefulset-like functionality without using statefulsets.
		lastDash := strings.LastIndexByte(pvcName, '-')
		if lastDash != -1 {
			statefulsetIDString := pvcName[lastDash+1:]
			statefulsetID, err := strconv.ParseUint(statefulsetIDString, 10, 32)
			if err == nil {
				// Offset by the statefulsetID, so we round-robin across zones
				index = uint32(statefulsetID)
				// We still hash the volume name, but only the prefix
				hashString = pvcName[:lastDash]

				// In the special case where it looks like `ClaimName-StatefulSetName-Id`,
				// hash only the StatefulSetName, so that different claims on the same StatefulSet
				// member end up in the same zone.
				// Note that StatefulSetName (and ClaimName) might themselves both have dashes.
				// We actually just take the portion after the final - of ClaimName-StatefulSetName.
				// For our purposes it doesn't much matter (just suboptimal spreading).
				lastDash := strings.LastIndexByte(hashString, '-')
				if lastDash != -1 {
					hashString = hashString[lastDash+1:]
				}

				klog.V(2).Infof("Detected StatefulSet-style volume name %q; index=%d", pvcName, index)
			}
		}

		// We hash the (base) volume name, so we don't bias towards the first N zones
		h := fnv.New32()
		h.Write([]byte(hashString))
		hash = h.Sum32()
	}

	return hash, index
}
