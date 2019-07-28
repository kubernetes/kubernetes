/*
Copyright 2018 The Kubernetes Authors.

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

// It's possible that kustomize's features will be vendored into
// the kubernetes/kubernetes repo and made available to kubectl
// commands, while at the same time the kustomize program will
// continue to exist as an independent CLI.  Vendoring snapshots
// would be taken just before a kubectl release.
//
// This creates a problem in that freestanding-kustomize depends on
// (for example):
//
//   https://github.com/kubernetes/apimachinery/
//       tree/master/pkg/util/yaml
//
// It vendors that package into
//   sigs.k8s.io/kustomize/vendor/k8s.io/apimachinery/
//
// Whereas kubectl-kustomize would have to depend on the "staging"
// version of this code, located at
//
//   https://github.com/kubernetes/kubernetes/
//       blob/master/staging/src/k8s.io/apimachinery/pkg/util/yaml
//
// which is "vendored" via symlinks:
//   k8s.io/kubernetes/vendor/k8s.io/apimachinery
// is a symlink to
//   ../../staging/src/k8s.io/apimachinery
//
// The staging version is the canonical, under-development
// version of the code that kubectl depends on, whereas the packages
// at kubernetes/apimachinery are periodic snapshots of staging made
// for outside tools to depend on.
//
// apimachinery isn't the only package that poses this problem, just
// using it as a specific example.
//
// The kubectl binary cannot vendor in kustomize code that in
// turn vendors in the non-staging packages.
//
// One way to fix some of this would be to copy code - a hard fork.
// This has all the problems associated with a hard forking.
//
// Another way would be to break the kustomize repo into three:
//
// (1) kustomize - repo with the main() function,
//     vendoring (2) and (3).
//
// (2) kustomize-libs - packages used by (1) with no
//     apimachinery dependence.
//
// (3) kustomize-k8sdeps - A thin code layer that depends
//     on (vendors) apimachinery to provide thin implementations
//     to interfaces used in (2).
//
// The kubectl repo would then vendor from (2) only, and have
// a local implementation of (3).  With that in mind, it's clear
// that (3) doesn't have to be a repo; the kustomize version of
// the thin layer can live directly in (1).
//
// This package is the code in (3), meant for kustomize.

package k8sdeps
