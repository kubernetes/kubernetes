package app

import (
	gcconfig "k8s.io/kubernetes/pkg/controller/garbagecollector/config"

	"k8s.io/kubernetes/cmd/kube-controller-manager/app/config"
)

func applyOpenShiftGCConfig(controllerManager *config.Config) error {
	// TODO make this configurable or discoverable.  This is going to prevent us from running the stock GC controller
	// IF YOU ADD ANYTHING TO THIS LIST, MAKE SURE THAT YOU UPDATE THEIR STRATEGIES TO PREVENT GC FINALIZERS
	//
	// DO NOT PUT CRDs into the list. apiexstension-apiserver does not implement GarbageCollectionPolicy
	// so the deletion of these will be blocked because of foregroundDeletion finalizer when foreground deletion strategy is specified.
	controllerManager.ComponentConfig.GarbageCollectorController.GCIgnoredResources = append(controllerManager.ComponentConfig.GarbageCollectorController.GCIgnoredResources,
		// explicitly disabled from GC for now - not enough value to track them
		gcconfig.GroupResource{Group: "oauth.openshift.io", Resource: "oauthclientauthorizations"},
		gcconfig.GroupResource{Group: "oauth.openshift.io", Resource: "oauthclients"},
		gcconfig.GroupResource{Group: "user.openshift.io", Resource: "groups"},
		gcconfig.GroupResource{Group: "user.openshift.io", Resource: "identities"},
		gcconfig.GroupResource{Group: "user.openshift.io", Resource: "users"},
		gcconfig.GroupResource{Group: "image.openshift.io", Resource: "images"},

		// virtual resource
		gcconfig.GroupResource{Group: "project.openshift.io", Resource: "projects"},
		// virtual and unwatchable resource, surfaced via rbac.authorization.k8s.io objects
		gcconfig.GroupResource{Group: "authorization.openshift.io", Resource: "clusterroles"},
		gcconfig.GroupResource{Group: "authorization.openshift.io", Resource: "clusterrolebindings"},
		gcconfig.GroupResource{Group: "authorization.openshift.io", Resource: "roles"},
		gcconfig.GroupResource{Group: "authorization.openshift.io", Resource: "rolebindings"},
		// these resources contain security information in their names, and we don't need to track them
		gcconfig.GroupResource{Group: "oauth.openshift.io", Resource: "oauthaccesstokens"},
		gcconfig.GroupResource{Group: "oauth.openshift.io", Resource: "oauthauthorizetokens"},
	)

	return nil
}
