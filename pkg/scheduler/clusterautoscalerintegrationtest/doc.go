/*
Copyright 2020 The Kubernetes Authors.

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

// *********** CLUSTER AUTOSCALER INTEGRATION TEST MODULE ********************
//
// Contents of this module mimic calls done by Cluster Autoscaler
// (https://github.com/kubernetes/autoscaler/tree/master/cluster-autoscaler) to
// scheduler codebase. The purpose of those is to quickly spot changes which may
// break interface between scheduler and cluster-autoscaler. We need this test
// because cluster-autoscaler directly calls scheduler internal interfaces
// (rather than calling out to public k8s APIs). It is needed to perform
// high-volume scheduling simulations but it renders scheduler <->
// cluster-autoscaler interface fragile.
//
// IF YOU NEED TO CHANGE THIS MODULE, REACH OUT TO CLUSTER-AUTOSCALER
// MAINTAINERS ON sig-autoscaling SLACK CHANNEL.
//
// ***************************************************************************

package clusterautoscalerintegrationtest

