/*
Copyright 2016 The Kubernetes Authors.

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

package kuberuntime

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"

	"github.com/stretchr/testify/assert"
	"testing"
)

func TestVerifyRunAsNonRoot(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "bar",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "foo",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
					Command:         []string{"testCommand"},
					WorkingDir:      "testWorkingDir",
				},
			},
		},
	}

	rootUser := int64(0)
	runAsNonRootTrue := true
	runAsNonRootFalse := false
	imageRootUser := int64(0)
	imageNonRootUser := int64(123)
	for _, test := range []struct {
		desc      string
		sc        *v1.SecurityContext
		imageUser int64
		fail      bool
	}{
		{
			desc:      "Pass if SecurityContext is not set",
			sc:        nil,
			imageUser: imageRootUser,
			fail:      false,
		},
		{
			desc: "Pass if RunAsNonRoot is not set",
			sc: &v1.SecurityContext{
				RunAsUser: &rootUser,
			},
			imageUser: imageRootUser,
			fail:      false,
		},
		{
			desc: "Pass if RunAsNonRoot is false (image user is root)",
			sc: &v1.SecurityContext{
				RunAsNonRoot: &runAsNonRootFalse,
			},
			imageUser: imageRootUser,
			fail:      false,
		},
		{
			desc: "Pass if RunAsNonRoot is false (RunAsUser is root)",
			sc: &v1.SecurityContext{
				RunAsNonRoot: &runAsNonRootFalse,
				RunAsUser:    &rootUser,
			},
			imageUser: imageNonRootUser,
			fail:      false,
		},
		{
			desc: "Fail if container's RunAsUser is root and RunAsNonRoot is true",
			sc: &v1.SecurityContext{
				RunAsNonRoot: &runAsNonRootTrue,
				RunAsUser:    &rootUser,
			},
			imageUser: imageNonRootUser,
			fail:      true,
		},
		{
			desc: "Fail if image's user is root and RunAsNonRoot is true",
			sc: &v1.SecurityContext{
				RunAsNonRoot: &runAsNonRootTrue,
			},
			imageUser: imageRootUser,
			fail:      true,
		},
	} {
		pod.Spec.Containers[0].SecurityContext = test.sc
		err := verifyRunAsNonRoot(pod, &pod.Spec.Containers[0], int64(0))
		if test.fail {
			assert.Error(t, err, test.desc)
		} else {
			assert.NoError(t, err, test.desc)
		}
	}
}
