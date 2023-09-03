//go:build windows
// +build windows

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

package kuberuntime

import (
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

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
					Image:           "windows",
					ImagePullPolicy: v1.PullIfNotPresent,
					Command:         []string{"testCommand"},
					WorkingDir:      "testWorkingDir",
				},
			},
		},
	}
	rootUser := "ContainerAdministrator"
	rootUserUppercase := "CONTAINERADMINISTRATOR"
	anyUser := "anyone"
	runAsNonRootTrue := true
	runAsNonRootFalse := false
	uid := int64(0)
	for _, test := range []struct {
		desc     string
		sc       *v1.SecurityContext
		uid      *int64
		username string
		fail     bool
	}{
		{
			desc:     "Pass if SecurityContext is not set",
			sc:       nil,
			username: rootUser,
			fail:     false,
		},
		{
			desc: "Pass if RunAsNonRoot is not set",
			sc: &v1.SecurityContext{
				RunAsNonRoot: nil,
			},
			username: rootUser,
			fail:     false,
		},
		{
			desc: "Pass if RunAsNonRoot is false (image user is root)",
			sc: &v1.SecurityContext{
				RunAsNonRoot: &runAsNonRootFalse,
			},
			username: rootUser,
			fail:     false,
		},
		{
			desc: "Pass if RunAsNonRoot is false (WindowsOptions RunAsUserName is root)",
			sc: &v1.SecurityContext{
				RunAsNonRoot: &runAsNonRootFalse,
				WindowsOptions: &v1.WindowsSecurityContextOptions{
					RunAsUserName: &rootUser,
				},
			},
			username: rootUser,
			fail:     false,
		},
		{
			desc: "Fail if container's RunAsUser is root and RunAsNonRoot is true",
			sc: &v1.SecurityContext{
				RunAsNonRoot: &runAsNonRootTrue,
				WindowsOptions: &v1.WindowsSecurityContextOptions{
					RunAsUserName: &rootUser,
				},
			},
			username: rootUser,
			fail:     true,
		},
		{
			desc: "Fail if container's RunAsUser is root (case-insensitive) and RunAsNonRoot is true",
			sc: &v1.SecurityContext{
				RunAsNonRoot: &runAsNonRootTrue,
				WindowsOptions: &v1.WindowsSecurityContextOptions{
					RunAsUserName: &rootUserUppercase,
				},
			},
			username: anyUser,
			fail:     true,
		},
		{
			desc: "Fail if image's user is root and RunAsNonRoot is true",
			sc: &v1.SecurityContext{
				RunAsNonRoot: &runAsNonRootTrue,
			},
			username: rootUser,
			fail:     true,
		},
		{
			desc: "Fail if image's user is root (case-insensitive) and RunAsNonRoot is true",
			sc: &v1.SecurityContext{
				RunAsNonRoot: &runAsNonRootTrue,
			},
			username: rootUserUppercase,
			fail:     true,
		},
		{
			desc: "Pass if image's user is non-root and RunAsNonRoot is true",
			sc: &v1.SecurityContext{
				RunAsNonRoot: &runAsNonRootTrue,
			},
			username: anyUser,
			fail:     false,
		},
		{
			desc: "Pass if container's user and image's user aren't set and RunAsNonRoot is true",
			sc: &v1.SecurityContext{
				// verifyRunAsNonRoot should ignore the RunAsUser, SELinuxOptions, and RunAsGroup options.
				RunAsUser:      &uid,
				SELinuxOptions: &v1.SELinuxOptions{},
				RunAsGroup:     &uid,
				RunAsNonRoot:   &runAsNonRootTrue,
			},
			fail: false,
		},
		{
			desc: "Pass if image's user is root, container's RunAsUser is not root and RunAsNonRoot is true",
			sc: &v1.SecurityContext{
				RunAsNonRoot: &runAsNonRootTrue,
				WindowsOptions: &v1.WindowsSecurityContextOptions{
					RunAsUserName: &anyUser,
				},
			},
			username: rootUser,
			fail:     false,
		},
		{
			desc: "Pass if image's user is root (case-insensitive), container's RunAsUser is not root and RunAsNonRoot is true",
			sc: &v1.SecurityContext{
				RunAsNonRoot: &runAsNonRootTrue,
				WindowsOptions: &v1.WindowsSecurityContextOptions{
					RunAsUserName: &anyUser,
				},
			},
			username: rootUserUppercase,
			fail:     false,
		},
	} {
		pod.Spec.Containers[0].SecurityContext = test.sc
		err := verifyRunAsNonRoot(pod, &pod.Spec.Containers[0], test.uid, test.username)
		if test.fail {
			assert.Error(t, err, test.desc)
		} else {
			assert.NoError(t, err, test.desc)
		}
	}
}
