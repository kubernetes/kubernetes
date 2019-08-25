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

package target_test

import (
	"testing"

	"sigs.k8s.io/kustomize/pkg/kusttest"
)

func TestPruneConfigMap(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/base")
	th.WriteK("/app/base", `
resources:
- deployment.yaml
- service.yaml
- secret.yaml

inventory:
  type: ConfigMap
  configMap:
    name: haha
    namespace: default

namePrefix: my-
namespace: default
`)
	th.WriteF("/app/base/deployment.yaml", `
apiVersion: apps/v1beta2
kind: Deployment
metadata:
  name: mysql
  labels:
    app: mysql
spec:
  selector:
    matchLabels:
      app: mysql
  strategy:
    type: Recreate
  template:
    metadata:
      labels:
        app: mysql
    spec:
      containers:
      - image: mysql:5.6
        name: mysql
        env:
        - name: MYSQL_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: pass
              key: password
        ports:
        - containerPort: 3306
          name: mysql
        volumeMounts:
        - name: mysql-persistent-storage
          mountPath: /var/lib/mysql
      volumes:
      - name: mysql-persistent-storage
        emptyDir: {}
`)
	th.WriteF("/app/base/service.yaml", `
apiVersion: v1
kind: Service
metadata:
  name: mmmysql
  labels:
    app: mysql
spec:
  ports:
    - port: 3306
  selector:
    app: mysql
`)
	th.WriteF("/app/base/secret.yaml", `
apiVersion: v1
kind: Secret
metadata:
  name: pass
type: Opaque
data:
  # Default password is "admin".
  password: YWRtaW4=
  username: jingfang
`)

	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	//nolint
	th.AssertActualEqualsExpected(m, `
apiVersion: apps/v1beta2
kind: Deployment
metadata:
  labels:
    app: mysql
  name: my-mysql
  namespace: default
spec:
  selector:
    matchLabels:
      app: mysql
  strategy:
    type: Recreate
  template:
    metadata:
      labels:
        app: mysql
    spec:
      containers:
      - env:
        - name: MYSQL_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              key: password
              name: my-pass
        image: mysql:5.6
        name: mysql
        ports:
        - containerPort: 3306
          name: mysql
        volumeMounts:
        - mountPath: /var/lib/mysql
          name: mysql-persistent-storage
      volumes:
      - emptyDir: {}
        name: mysql-persistent-storage
---
apiVersion: v1
kind: Service
metadata:
  labels:
    app: mysql
  name: my-mmmysql
  namespace: default
spec:
  ports:
  - port: 3306
  selector:
    app: mysql
---
apiVersion: v1
data:
  password: YWRtaW4=
  username: jingfang
kind: Secret
metadata:
  name: my-pass
  namespace: default
type: Opaque
---
apiVersion: v1
kind: ConfigMap
metadata:
  annotations:
    kustomize.config.k8s.io/Inventory: '{"current":{"apps_v1beta2_Deployment|default|my-mysql":null,"~G_v1_Secret|default|my-pass":[{"group":"apps","version":"v1beta2","kind":"Deployment","name":"my-mysql","namespace":"default"}],"~G_v1_Service|default|my-mmmysql":null}}'
    kustomize.config.k8s.io/InventoryHash: kd67f7ht8t
  name: haha
  namespace: default
`)
}
