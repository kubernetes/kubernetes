# cloud-controller-manager/example

This directory provides an example of how to leverage CCM extension mechanism.

## Purpose

Begin with 1.20, all cloud providers should not copy over or vendor in `k8s.io/kubernetes/cmd/cloud-controller-manager`. Inside this directory, an example is included to demonstrate how to leverage CCM extension mechanism to add a controller.
Please refer to `k8s.io/cloud-provider/sample` if you do not have the requirement of adding/deleting controllers in CCM.

## Things you should NOT do

1. Vendor in `k8s.io/cmd/cloud-controller-manager`.
2. Directly modify anything under `k8s.io/cmd/cloud-controller-manager` in this repo. 
3. Make specific cloud provider changes here.
