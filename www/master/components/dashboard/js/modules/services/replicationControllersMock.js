(function() {
  'use strict';

  angular.module('replicationControllers', [])
      .service('replicationControllerService', ReplicationControllerDataService);

  /**
   * Replication Controller DataService
   * Mock async data service.
   *
   * @returns {{loadAll: Function}}
   * @constructor
   */
  function ReplicationControllerDataService($q) {
    var replicationControllers = {
    "kind": "List",
    "apiVersion": "v1",
    "metadata": {},
    "items": [
        {
            "kind": "ReplicationController",
            "apiVersion": "v1",
            "metadata": {
                "name": "redis-master",
                "namespace": "default",
                "selfLink": "/api/v1/namespaces/default/replicationcontrollers/redis-master",
                "uid": "f12969e0-ff77-11e4-8f2d-080027213276",
                "resourceVersion": "28",
                "creationTimestamp": "2015-05-21T05:12:14Z",
                "labels": {
                    "name": "redis-master"
                }
            },
            "spec": {
                "replicas": 1,
                "selector": {
                    "name": "redis-master"
                },
                "template": {
                    "metadata": {
                        "creationTimestamp": null,
                        "labels": {
                            "name": "redis-master"
                        }
                    },
                    "spec": {
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
                        "serviceAccount": ""
                    }
                }
            },
            "status": {
                "replicas": 1
            }
        }
    ]};

    // Uses promises
    return {
      loadAll: function() {
        // Simulate async call
        return $q.when(replicationControllers);
      }
    };
  }

})();
