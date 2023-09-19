package kubelet

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/nodestatus"
)

type hostState struct {
	validators []validator
}

func newHostState() *hostState {
	return &hostState{}
}

func (s *hostState) addValidator(validator validator) {
	s.validators = append(s.validators, validator)
}

func (s *hostState) hostErrors() error {
	var errs []error
	for _, v := range s.validators {
		if err := v.validate(); err != nil {
			errs = append(errs, err)
		}
	}

	return errors.NewAggregate(errs)
}

type validator interface {
	validate() error
}

type nodeIPsValidator struct {
	nodeIPs *nodestatus.NodeIPsState
}

func (n *nodeIPsValidator) validate() error {
	nodeIP, secondaryNodeIP := n.nodeIPs.NodeIP, n.nodeIPs.SecondaryNodeIP
	if n.nodeIPs.NodeIPSpecified {
		if err := validateNodeIP(nodeIP); err != nil {
			return fmt.Errorf("failed to validate nodeIP: %v", err)
		}
		klog.V(4).InfoS("Using node IP", "IP", nodeIP.String())
	}
	if n.nodeIPs.SecondaryNodeIPSpecified {
		if err := validateNodeIP(secondaryNodeIP); err != nil {
			return fmt.Errorf("failed to validate secondaryNodeIP: %v", err)
		}
		klog.V(4).InfoS("Using secondary node IP", "IP", secondaryNodeIP.String())
	}
	return nil
}

func newNodeIPsValidator(nodeIPsState *nodestatus.NodeIPsState) *nodeIPsValidator {
	return &nodeIPsValidator{
		nodeIPs: nodeIPsState,
	}
}
