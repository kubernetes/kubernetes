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

package csitranslation

import (
	"fmt"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/csi-translation-lib/plugins"
)

var (
	kubernetesBetaTopologyLabels = map[string]string{
		v1.LabelFailureDomainBetaZone:   "us-east-1a",
		v1.LabelFailureDomainBetaRegion: "us-east-1",
	}
	kubernetesGATopologyLabels = map[string]string{
		v1.LabelTopologyZone:   "us-east-1a",
		v1.LabelTopologyRegion: "us-east-1",
	}
	regionalBetaPDLabels = map[string]string{
		v1.LabelFailureDomainBetaZone: "europe-west1-b__europe-west1-c",
	}
	regionalGAPDLabels = map[string]string{
		v1.LabelTopologyZone: "europe-west1-b__europe-west1-c",
	}
)

func TestTranslationStability(t *testing.T) {
	testCases := []struct {
		name string
		pv   *v1.PersistentVolume
	}{

		{
			name: "GCE PD PV Source",
			pv: &v1.PersistentVolume{
				Spec: v1.PersistentVolumeSpec{
					PersistentVolumeSource: v1.PersistentVolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName:    "test-disk",
							FSType:    "ext4",
							Partition: 0,
							ReadOnly:  false,
						},
					},
				},
			},
		},
		{
			name: "AWS EBS PV Source",
			pv: &v1.PersistentVolume{
				Spec: v1.PersistentVolumeSpec{
					PersistentVolumeSource: v1.PersistentVolumeSource{
						AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
							VolumeID:  "vol01",
							FSType:    "ext3",
							Partition: 1,
							ReadOnly:  true,
						},
					},
				},
			},
		},
	}
	for _, test := range testCases {
		ctl := New()
		t.Logf("Testing %v", test.name)
		csiSource, err := ctl.TranslateInTreePVToCSI(test.pv)
		if err != nil {
			t.Errorf("Error when translating to CSI: %v", err)
		}
		newPV, err := ctl.TranslateCSIPVToInTree(csiSource)
		if err != nil {
			t.Errorf("Error when translating CSI Source to in tree volume: %v", err)
		}
		if !reflect.DeepEqual(newPV, test.pv) {
			t.Errorf("Volumes after translation and back not equal:\n\nOriginal Volume: %#v\n\nRound-trip Volume: %#v", test.pv, newPV)
		}
	}
}

func TestTopologyTranslation(t *testing.T) {
	testCases := []struct {
		name                 string
		key                  string
		pv                   *v1.PersistentVolume
		expectedNodeAffinity *v1.VolumeNodeAffinity
	}{
		{
			name:                 "GCE PD with beta zone labels",
			key:                  plugins.GCEPDTopologyKey,
			pv:                   makeGCEPDPV(kubernetesBetaTopologyLabels, nil /*topology*/),
			expectedNodeAffinity: makeNodeAffinity(false /*multiTerms*/, plugins.GCEPDTopologyKey, "us-east-1a"),
		},
		{
			name:                 "GCE PD with GA kubernetes zone labels",
			key:                  plugins.GCEPDTopologyKey,
			pv:                   makeGCEPDPV(kubernetesGATopologyLabels, nil /*topology*/),
			expectedNodeAffinity: makeNodeAffinity(false /*multiTerms*/, plugins.GCEPDTopologyKey, "us-east-1a"),
		},
		{
			name:                 "GCE PD with existing topology (beta keys)",
			pv:                   makeGCEPDPV(nil /*labels*/, makeTopology(v1.LabelFailureDomainBetaZone, "us-east-2a")),
			expectedNodeAffinity: makeNodeAffinity(false /*multiTerms*/, plugins.GCEPDTopologyKey, "us-east-2a"),
		},
		{
			name:                 "GCE PD with existing topology (CSI keys)",
			key:                  plugins.GCEPDTopologyKey,
			pv:                   makeGCEPDPV(nil /*labels*/, makeTopology(plugins.GCEPDTopologyKey, "us-east-2a")),
			expectedNodeAffinity: makeNodeAffinity(false /*multiTerms*/, plugins.GCEPDTopologyKey, "us-east-2a"),
		},
		{
			name:                 "GCE PD with zone labels and topology",
			pv:                   makeGCEPDPV(kubernetesBetaTopologyLabels, makeTopology(v1.LabelFailureDomainBetaZone, "us-east-2a")),
			expectedNodeAffinity: makeNodeAffinity(false /*multiTerms*/, plugins.GCEPDTopologyKey, "us-east-2a"),
		},
		{
			name:                 "GCE PD with regional zones",
			key:                  plugins.GCEPDTopologyKey,
			pv:                   makeGCEPDPV(regionalBetaPDLabels, nil /*topology*/),
			expectedNodeAffinity: makeNodeAffinity(false /*multiTerms*/, plugins.GCEPDTopologyKey, "europe-west1-b", "europe-west1-c"),
		},
		{
			name:                 "GCE PD with regional topology",
			key:                  plugins.GCEPDTopologyKey,
			pv:                   makeGCEPDPV(nil /*labels*/, makeTopology(v1.LabelTopologyZone, "europe-west1-b", "europe-west1-c")),
			expectedNodeAffinity: makeNodeAffinity(false /*multiTerms*/, plugins.GCEPDTopologyKey, "europe-west1-b", "europe-west1-c"),
		},
		{
			name:                 "GCE PD with Beta regional zone and topology",
			key:                  plugins.GCEPDTopologyKey,
			pv:                   makeGCEPDPV(regionalBetaPDLabels, makeTopology(v1.LabelFailureDomainBetaZone, "europe-west1-f", "europe-west1-g")),
			expectedNodeAffinity: makeNodeAffinity(false /*multiTerms*/, plugins.GCEPDTopologyKey, "europe-west1-f", "europe-west1-g"),
		},
		{
			name:                 "GCE PD with GA regional zone and topology",
			key:                  plugins.GCEPDTopologyKey,
			pv:                   makeGCEPDPV(regionalGAPDLabels, makeTopology(v1.LabelTopologyZone, "europe-west1-f", "europe-west1-g")),
			expectedNodeAffinity: makeNodeAffinity(false /*multiTerms*/, plugins.GCEPDTopologyKey, "europe-west1-f", "europe-west1-g"),
		},
		{
			name: "GCE PD with multiple node selector terms",
			key:  plugins.GCEPDTopologyKey,
			pv: makeGCEPDPVMultTerms(
				nil, /*labels*/
				makeTopology(v1.LabelTopologyZone, "europe-west1-f"),
				makeTopology(v1.LabelTopologyZone, "europe-west1-g")),
			expectedNodeAffinity: makeNodeAffinity(
				true, /*multiTerms*/
				plugins.GCEPDTopologyKey, "europe-west1-f", "europe-west1-g"),
		},
		// EBS test cases: test mostly topology key, i.e., don't repeat testing done with GCE
		{
			name:                 "AWS EBS with beta zone labels",
			pv:                   makeAWSEBSPV(kubernetesBetaTopologyLabels, nil /*topology*/),
			expectedNodeAffinity: makeNodeAffinity(false /*multiTerms*/, plugins.AWSEBSTopologyKey, "us-east-1a"),
		},
		{
			name:                 "AWS EBS with beta zone labels and topology",
			pv:                   makeAWSEBSPV(kubernetesBetaTopologyLabels, makeTopology(v1.LabelFailureDomainBetaZone, "us-east-2a")),
			expectedNodeAffinity: makeNodeAffinity(false /*multiTerms*/, plugins.AWSEBSTopologyKey, "us-east-2a"),
		},
		{
			name:                 "AWS EBS with GA zone labels",
			pv:                   makeAWSEBSPV(kubernetesGATopologyLabels, nil /*topology*/),
			expectedNodeAffinity: makeNodeAffinity(false /*multiTerms*/, plugins.AWSEBSTopologyKey, "us-east-1a"),
		},
		{
			name:                 "AWS EBS with GA zone labels and topology",
			pv:                   makeAWSEBSPV(kubernetesGATopologyLabels, makeTopology(v1.LabelTopologyZone, "us-east-2a")),
			expectedNodeAffinity: makeNodeAffinity(false /*multiTerms*/, plugins.AWSEBSTopologyKey, "us-east-2a"),
		},
	}

	for _, test := range testCases {
		ctl := New()
		t.Logf("Testing %v", test.name)

		// Translate to CSI PV and check translated node affinity
		newCSIPV, err := ctl.TranslateInTreePVToCSI(test.pv)
		if err != nil {
			t.Errorf("Error when translating to CSI: %v", err)
		}

		nodeAffinity := newCSIPV.Spec.NodeAffinity
		if !reflect.DeepEqual(nodeAffinity, test.expectedNodeAffinity) {
			t.Errorf("Expected node affinity %v, got %v", *test.expectedNodeAffinity, *nodeAffinity)
		}

		// Translate back to in-tree and make sure node affinity has been removed
		newInTreePV, err := ctl.TranslateCSIPVToInTree(newCSIPV)
		if err != nil {
			t.Errorf("Error when translating to in-tree: %v", err)
		}

		// For now, non-pd cloud should stay the old behavior which is still have the CSI topology.
		if test.key != "" {
			nodeAffinity = newInTreePV.Spec.NodeAffinity
			if plugins.TopologyKeyExist(test.key, nodeAffinity) {
				t.Errorf("Expected node affinity key %v being removed, got %v", test.key, *nodeAffinity)
			}
			// verify that either beta or GA kubernetes topology key should exist
			if !(plugins.TopologyKeyExist(v1.LabelFailureDomainBetaZone, nodeAffinity) || plugins.TopologyKeyExist(v1.LabelTopologyZone, nodeAffinity)) {
				t.Errorf("Expected node affinity kuberenetes topology label exist, got %v", *nodeAffinity)
			}
		} else {
			nodeAffinity := newCSIPV.Spec.NodeAffinity
			if !reflect.DeepEqual(nodeAffinity, test.expectedNodeAffinity) {
				t.Errorf("Expected node affinity %v, got %v", *test.expectedNodeAffinity, *nodeAffinity)
			}
		}
	}
}

func makePV(labels map[string]string, topology *v1.NodeSelectorRequirement) *v1.PersistentVolume {
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Labels: labels,
		},
		Spec: v1.PersistentVolumeSpec{},
	}

	if topology != nil {
		pv.Spec.NodeAffinity = &v1.VolumeNodeAffinity{
			Required: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{
					{MatchExpressions: []v1.NodeSelectorRequirement{*topology}},
				},
			},
		}
	}

	return pv
}

func makeGCEPDPV(labels map[string]string, topology *v1.NodeSelectorRequirement) *v1.PersistentVolume {
	pv := makePV(labels, topology)
	pv.Spec.PersistentVolumeSource = v1.PersistentVolumeSource{
		GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
			PDName:    "test-disk",
			FSType:    "ext4",
			Partition: 0,
			ReadOnly:  false,
		},
	}
	return pv
}

func makeGCEPDPVMultTerms(labels map[string]string, topologies ...*v1.NodeSelectorRequirement) *v1.PersistentVolume {
	pv := makeGCEPDPV(labels, topologies[0])
	for _, topology := range topologies[1:] {
		pv.Spec.NodeAffinity.Required.NodeSelectorTerms = append(
			pv.Spec.NodeAffinity.Required.NodeSelectorTerms,
			v1.NodeSelectorTerm{
				MatchExpressions: []v1.NodeSelectorRequirement{*topology},
			},
		)
	}
	return pv
}

func makeAWSEBSPV(labels map[string]string, topology *v1.NodeSelectorRequirement) *v1.PersistentVolume {
	pv := makePV(labels, topology)
	pv.Spec.PersistentVolumeSource = v1.PersistentVolumeSource{
		AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
			VolumeID:  "vol01",
			FSType:    "ext3",
			Partition: 1,
			ReadOnly:  true,
		},
	}
	return pv
}

func makeNodeAffinity(multiTerms bool, key string, values ...string) *v1.VolumeNodeAffinity {
	nodeAffinity := &v1.VolumeNodeAffinity{
		Required: &v1.NodeSelector{
			NodeSelectorTerms: []v1.NodeSelectorTerm{
				{
					MatchExpressions: []v1.NodeSelectorRequirement{
						{
							Key:      key,
							Operator: v1.NodeSelectorOpIn,
							Values:   values,
						},
					},
				},
			},
		},
	}

	// If multiple terms is NOT requested, return a single term with all values
	if !multiTerms {
		return nodeAffinity
	}

	// Otherwise return multiple terms, each one with a single value
	nodeAffinity.Required.NodeSelectorTerms[0].MatchExpressions[0].Values = values[:1] // If values=[1,2,3], overwrite with [1]
	for _, value := range values[1:] {
		term := v1.NodeSelectorTerm{
			MatchExpressions: []v1.NodeSelectorRequirement{
				{
					Key:      key,
					Operator: v1.NodeSelectorOpIn,
					Values:   []string{value},
				},
			},
		}
		nodeAffinity.Required.NodeSelectorTerms = append(nodeAffinity.Required.NodeSelectorTerms, term)
	}

	return nodeAffinity
}

func makeTopology(key string, values ...string) *v1.NodeSelectorRequirement {
	return &v1.NodeSelectorRequirement{
		Key:      key,
		Operator: v1.NodeSelectorOpIn,
		Values:   values,
	}
}

func TestTranslateInTreeInlineVolumeToCSINameUniqueness(t *testing.T) {
	for driverName := range inTreePlugins {
		t.Run(driverName, func(t *testing.T) {
			ctl := New()
			vs1, err := generateUniqueVolumeSource(driverName)
			if err != nil {
				t.Fatalf("Couldn't generate random source: %v", err)
			}
			pv1, err := ctl.TranslateInTreeInlineVolumeToCSI(&v1.Volume{
				VolumeSource: vs1,
			}, "")
			if err != nil {
				t.Fatalf("Error when translating to CSI: %v", err)
			}
			vs2, err := generateUniqueVolumeSource(driverName)
			if err != nil {
				t.Fatalf("Couldn't generate random source: %v", err)
			}
			pv2, err := ctl.TranslateInTreeInlineVolumeToCSI(&v1.Volume{
				VolumeSource: vs2,
			}, "")
			if err != nil {
				t.Fatalf("Error when translating to CSI: %v", err)
			}
			if pv1 == nil || pv2 == nil {
				t.Fatalf("Did not expect either pv1: %v or pv2: %v to be nil", pv1, pv2)
			}
			if pv1.Name == pv2.Name {
				t.Errorf("PV name %s not sufficiently unique for different volumes", pv1.Name)
			}
		})

	}
}

func generateUniqueVolumeSource(driverName string) (v1.VolumeSource, error) {
	switch driverName {
	case plugins.GCEPDDriverName:
		return v1.VolumeSource{
			GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
				PDName: string(uuid.NewUUID()),
			},
		}, nil
	case plugins.AWSEBSDriverName:
		return v1.VolumeSource{
			AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
				VolumeID: string(uuid.NewUUID()),
			},
		}, nil

	case plugins.AzureDiskDriverName:
		return v1.VolumeSource{
			AzureDisk: &v1.AzureDiskVolumeSource{
				DiskName:    string(uuid.NewUUID()),
				DataDiskURI: string(uuid.NewUUID()),
			},
		}, nil
	case plugins.AzureFileDriverName:
		return v1.VolumeSource{
			AzureFile: &v1.AzureFileVolumeSource{
				SecretName: string(uuid.NewUUID()),
				ShareName:  string(uuid.NewUUID()),
			},
		}, nil
	case plugins.VSphereDriverName:
		return v1.VolumeSource{
			VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
				VolumePath: " [vsanDatastore] 6785a85e-268e-6352-a2e8-02008b7afadd/kubernetes-dynamic-pvc-" + string(uuid.NewUUID()+".vmdk"),
				FSType:     "ext4",
			},
		}, nil
	case plugins.PortworxDriverName:
		return v1.VolumeSource{
			PortworxVolume: &v1.PortworxVolumeSource{
				VolumeID: string(uuid.NewUUID()),
			},
		}, nil
	case plugins.RBDDriverName:
		return v1.VolumeSource{
			RBD: &v1.RBDVolumeSource{
				RBDImage: string(uuid.NewUUID()),
			},
		}, nil
	default:
		return v1.VolumeSource{}, fmt.Errorf("couldn't find logic for driver: %v", driverName)
	}
}

func TestPluginNameMappings(t *testing.T) {
	testCases := []struct {
		name             string
		inTreePluginName string
		csiPluginName    string
	}{
		{
			name:             "GCE PD plugin name",
			inTreePluginName: "kubernetes.io/gce-pd",
			csiPluginName:    "pd.csi.storage.gke.io",
		},
		{
			name:             "AWS EBS plugin name",
			inTreePluginName: "kubernetes.io/aws-ebs",
			csiPluginName:    "ebs.csi.aws.com",
		},
		{
			name:             "RBD plugin name",
			inTreePluginName: "kubernetes.io/rbd",
			csiPluginName:    "rbd.csi.ceph.com",
		},
	}
	for _, test := range testCases {
		t.Logf("Testing %v", test.name)
		ctl := New()
		csiPluginName, err := ctl.GetCSINameFromInTreeName(test.inTreePluginName)
		if err != nil {
			t.Errorf("Error when mapping In-tree plugin name to CSI plugin name %s", err)
		}
		if !ctl.IsMigratedCSIDriverByName(csiPluginName) {
			t.Errorf("%s expected to supersede an In-tree plugin", csiPluginName)
		}
		inTreePluginName, err := ctl.GetInTreeNameFromCSIName(csiPluginName)
		if err != nil {
			t.Errorf("Error when mapping CSI plugin name to In-tree plugin name %s", err)
		}
		if !ctl.IsMigratableIntreePluginByName(inTreePluginName) {
			t.Errorf("%s expected to be migratable to a CSI name", inTreePluginName)
		}
		if inTreePluginName != test.inTreePluginName || csiPluginName != test.csiPluginName {
			t.Errorf("CSI plugin name and In-tree plugin name do not map to each other: [%s => %s], [%s => %s]", test.csiPluginName, inTreePluginName, test.inTreePluginName, csiPluginName)
		}
	}
}

// TODO: test for not modifying the original PV.
