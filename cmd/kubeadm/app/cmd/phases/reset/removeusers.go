package phases

import (
	"errors"
	"fmt"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/controlplane"
)

// NewRemoveETCDMemberPhase creates a kubeadm workflow phase for remove-etcd-member
func NewRemoveUsersPhase() workflow.Phase {
	return workflow.Phase{
		Name:         "remove-users",
		Short:        "Remove users and groups created for control-plane.",
		Long:         "Remove users and groups created for control-plane.",
		Run:          runRemoveUsersPhase,
		InheritFlags: []string{},
	}
}

func runRemoveUsersPhase(c workflow.RunData) error {
	if err := controlplane.RemoveUsersAndGroups(); err != nil {
		return errors.New(fmt.Sprintf("Failed to remove control-plane users: %v", err))
	}

	return nil
}
