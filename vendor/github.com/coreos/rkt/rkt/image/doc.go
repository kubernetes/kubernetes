// Copyright 2015 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package image implements finding images in the store and fetching
// them from local or remote locations. The only API exposed are
// Finder and Fetcher - all their fields are also exposed (see action
// in common.go).
//
// Hacking docs:
//
// Documentation of a specific component is in its related file. Below
// is a short top-down documentation about relations between various
// types in this package.
//
// Finder uses Fetcher to get the image from remote if it could not
// find one in the store.
//
// Fetcher delegates its work to a specific fetcher. Specific fetchers
// currently available are fileFetcher, dockerFetcher, nameFetcher and
// httpFetcher.
//
// fileFetcher gets the images from a local filesystem. Fetcher uses
// it when the image reference is either a path (relative or absolute)
// or a file:// URL. In the latter case fileFetcher receives only the
// Path part of the URL. fileFetcher also uses validator to verify the
// image.
//
// dockerFetcher gets the images from docker registries. Fetcher uses
// it when the image reference is a docker:// URL.
//
// httpFetcher gets the images from http:// or https:// URLs. It uses
// httpOps for doing all the downloading, and validator to verify the
// image.
//
// nameFetcher gets the images via a discovery process. Fetcher uses
// it when the image reference is an image name. nameFetcher does the
// discovery and then uses httpOps for doing all the downloading, and
// validator to verify the image.
//
// validator checks various things in the downloaded image. It can
// check whether the downloaded file is a valid image, so it can get
// an image manifest from it. It also can check if the image has an
// expected name or a if a signature of the image can be trusted.
//
// httpOps does the downloading of the images and signatures. It also
// provides a fetcher for remoteAscFetcher to download the signature.
// For the downloading process itself it uses downloader and
// resumableSession.
//
// asc is used to get the signature either from a local filesystem or
// from some remote location. It also provides an ascFetcher interface
// and two implementations - localAscFetcher and remoteAscFetcher. The
// former is standalone, the latter needs a function that does the
// heavy-lifting (actually the downloading, currently provided by
// httpOps).
//
// resumableSession is an implementation of a downloadSession
// interface, so it is used together with downloader.
//
// downloader also provides a downloadSession interface and uses its
// implementation to, uh, download stuff. It also provides a
// dead-simple implementation of downloadSession -
// defaultDownloadSession.
//
// There are also various functions and types in common.go and io.go,
// which are used by the types listed above and their functions.
package image
