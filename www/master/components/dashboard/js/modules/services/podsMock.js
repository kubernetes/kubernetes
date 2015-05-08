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
      "kind": "PodList",
      "creationTimestamp": null,
      "selfLink": "/api/v1beta1/pods",
      "resourceVersion": 166552,
      "apiVersion": "v1beta1",
      "items": [{
        "id": "hello",
        "uid": "0fe3644e-ab53-11e4-8ae8-061695c59fcf",
        "creationTimestamp": "2015-02-03T03:16:36Z",
        "selfLink": "/api/v1beta1/pods/hello?namespace=default",
        "resourceVersion": 466,
        "namespace": "default",
        "labels": {"environment": "testing", "name": "hello"},
        "desiredState": {
          "manifest": {
            "version": "v1beta2",
            "id": "",
            "volumes": null,
            "containers": [{
              "name": "hello",
              "image": "quay.io/kelseyhightower/hello",
              "ports": [{"hostPort": 80, "containerPort": 80, "protocol": "TCP"}],
              "imagePullPolicy": "PullIfNotPresent"
            }],
            "restartPolicy": {"always": {}},
            "dnsPolicy": "ClusterFirst"
          }
        },
        "currentState": {
          "manifest": {"version": "", "id": "", "volumes": null, "containers": null, "restartPolicy": {}},
          "status": "Running",
          "host": "172.31.12.204",
          "podIP": "10.244.73.2",
          "info": {
            "hello": {
              "state": {"running": {"startedAt": "2015-02-03T03:16:51Z"}},
              "restartCount": 0,
              "image": "quay.io/kelseyhightower/hello",
              "containerID": "docker://96ade8ff30a44c4489969eaf343a7899317671b07a9766ecd0963e9b41501256"
            },
            "net": {
              "state": {"running": {"startedAt": "2015-02-03T03:16:41Z"}},
              "restartCount": 0,
              "podIP": "10.244.73.2",
              "image": "kubernetes/pause:latest",
              "containerID": "docker://93d32603cafbff7165dadb1d4527899c24246bca2f5e6770b8297fd3721b272c"
            }
          }
        }
      }]
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
