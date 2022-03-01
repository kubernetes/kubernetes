# cloud-provider/sample

This directory provides sample code about how all cloud providers should leverage CCM begin at 1.20.

## Purpose

Begin with 1.20, all cloud providers should not copy over or vendor in `k8s.io/kubernetes/cmd/cloud-controller-manager`. Inside this directory, some sample code will be provided to demonstrate how cloud providers should leverage cloud-controller-manager. 

## Steps cloud providers should follow

1. Have your external repo under k8s.io. e.g. `k8s.io/cloud-provider-<provider>`
2. Create `main.go` file under your external repo CCM directory. Please refer to `basic_main.go` for a minimum working sample.
Note: If you have a requirement of adding/deleting controllers within CCM, please refer to `k8s.io/kubernetes/cmd/cloud-controller-manager/main.go` for extra details.
3. Build/release CCM from your external repo. For existing cloud providers, the option to import legacy providers from `k8s.io/legacy-cloud-provider/<provider>` is still available.

## Things you should NOT do

 1. Vendor in `k8s.io/cmd/cloud-controller-manager`.
 2. Directly modify anything under `vendor/k8s.io/cloud-provider/sample` in this repo. Those are driven from `k8s.io/kubernetes/staging/src/k8s.io/cloud-provider/sample`.
 3. Make specific cloud provider changes in sample files.
