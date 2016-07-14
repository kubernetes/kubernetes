
/*
Copyright 2016 The Kubernetes Authors All.
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

package images

type ImageManager struct {
	recorder record.EventRecorder	
	runtime  container.Runtime
	backOff  *flowcontrol.Backoff
	imagePuller imagePuller
}

func NewImageManager(recorder record.EventRecorder, runtime Runtime, imageBackOff *flowcontrol.Backoff, serialized bool) ImageManager {
	var imagePuller imagePuller
	if serialized {
		imagePuller = NewSerializedImagePuller(recorder, runtime, imageBackOff)
	} else {
		imagePuller = NewParallelImagePuller(recorder, runtime, imageBackOff)
	}
	return &imageManager{
 		recorder: 		recorder,
 		runtime:  		runtime,
 		backOff:  		backOff,
 		imagePuller:   imagePuller,
 	}
}

func (*) EnsureImageExists(pod *api.Pod, container *api.Container, pullSecrets []api.Secret) (error, string) {
	return imagePuller.pullImage(pod , container pullSecrets)
}