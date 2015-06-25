(function() {
  'use strict';

  angular.module('pods', []).service('podService', PodDataService);

  /**
   * Pod DataService
   * Mock async data service.
   *
   * @returns {{loadAll: Function}}
   * @constructor
   */
  function PodDataService($q) {
    var pods = {
    "kind": "Pod",
    "apiVersion": "v1",
    "metadata": {
        "name": "redis-master-c0r1n",
        "generateName": "redis-master-",
        "namespace": "default",
        "selfLink": "/api/v1/namespaces/default/pods/redis-master-c0r1n",
        "uid": "f12ddfaf-ff77-11e4-8f2d-080027213276",
        "resourceVersion": "39",
        "creationTimestamp": "2015-05-21T05:12:14Z",
        "labels": {
            "name": "redis-master"
        },
        "annotations": {
            "kubernetes.io/created-by": "{\"kind\":\"SerializedReference\",\"apiVersion\":\"v1\",\"reference\":{\"kind\":\"ReplicationController\",\"namespace\":\"default\",\"name\":\"redis-master\",\"uid\":\"f12969e0-ff77-11e4-8f2d-080027213276\",\"apiVersion\":\"v1\",\"resourceVersion\":\"26\"}}"
        }
    },
    "spec": {
        "volumes": [
            {
                "name": "default-token-zb4rq",
                "secret": {
                    "secretName": "default-token-zb4rq"
                }
            }
        ],
        "containers": [
            {
                "name": "master",
                "image": "redis",
                "ports": [
                    {
                        "containerPort": 6379,
                        "protocol": "TCP"
                    }
                ],
                "resources": {},
                "volumeMounts": [
                    {
                        "name": "default-token-zb4rq",
                        "readOnly": true,
                        "mountPath": "/var/run/secrets/kubernetes.io/serviceaccount"
                    }
                ],
                "terminationMessagePath": "/dev/termination-log",
                "imagePullPolicy": "IfNotPresent",
                "capabilities": {},
                "securityContext": {
                    "capabilities": {},
                    "privileged": false
                }
            }
        ],
        "restartPolicy": "Always",
        "dnsPolicy": "ClusterFirst",
        "serviceAccount": "default",
        "host": "127.0.0.1"
    },
    "status": {
        "phase": "Running",
        "Condition": [
            {
                "type": "Ready",
                "status": "True"
            }
        ],
        "hostIP": "127.0.0.1",
        "podIP": "172.17.0.1",
        "startTime": "2015-05-21T05:12:14Z",
        "containerStatuses": [
            {
                "name": "master",
                "state": {
                    "running": {
                        "startedAt": "2015-05-21T05:12:14Z"
                    }
                },
                "lastState": {},
                "ready": true,
                "restartCount": 0,
                "image": "redis",
                "imageID": "docker://95af5842ddb9b03f7c6ec7601e65924cec516fcedd7e590ae31660057085cf67",
                "containerID": "docker://ae2a1e0a91a8b1015191a0b8e2ce8c55a86fb1a9a2b1e8e3b29430c9d93c8c09"
            }
        ]
    }
};

    // Uses promises
    return {
      loadAll: function() {
        // Simulate async call
        return $q.when(pods);
      }
    };
  }

})();
