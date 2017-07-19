package options

import (
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
)

var featureGates = map[utilfeature.Feature]utilfeature.FeatureSpec{
	features.DynamicKubeletConfig:                        {Default: false, PreRelease: utilfeature.Alpha},
	features.ExperimentalHostUserNamespaceDefaultingGate: {Default: false, PreRelease: utilfeature.Beta},
	features.ExperimentalCriticalPodAnnotation:           {Default: false, PreRelease: utilfeature.Alpha},
	features.Accelerators:                                {Default: false, PreRelease: utilfeature.Alpha},
	features.RotateKubeletServerCertificate:              {Default: false, PreRelease: utilfeature.Alpha},
	features.RotateKubeletClientCertificate:              {Default: false, PreRelease: utilfeature.Alpha},
	features.LocalStorageCapacityIsolation:               {Default: false, PreRelease: utilfeature.Alpha},
	features.DebugContainers:                             {Default: false, PreRelease: utilfeature.Alpha},
}
