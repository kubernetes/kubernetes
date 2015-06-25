(function() {
  'use strict';

  angular.module('services', []).service('serviceService', ServiceDataService);

  /**
   * Service DataService
   * Mock async data service.
   *
   * @returns {{loadAll: Function}}
   * @constructor
   */
  function ServiceDataService($q) {
    var services = {
    "kind": "List",
    "apiVersion": "v1",
    "metadata": {},
    "items": [
        {
            "kind": "Service",
            "apiVersion": "v1",
            "metadata": {
                "name": "kubernetes",
                "namespace": "default",
                "selfLink": "/api/v1/namespaces/default/services/kubernetes",
                "resourceVersion": "6",
                "creationTimestamp": null,
                "labels": {
                    "component": "apiserver",
                    "provider": "kubernetes"
                }
            },
            "spec": {
                "ports": [
                    {
                        "protocol": "TCP",
                        "port": 443,
                        "targetPort": 443
                    }
                ],
                "portalIP": "10.0.0.2",
                "sessionAffinity": "None"
            },
            "status": {}
        },
        {
            "kind": "Service",
            "apiVersion": "v1",
            "metadata": {
                "name": "kubernetes-ro",
                "namespace": "default",
                "selfLink": "/api/v1/namespaces/default/services/kubernetes-ro",
                "resourceVersion": "8",
                "creationTimestamp": null,
                "labels": {
                    "component": "apiserver",
                    "provider": "kubernetes"
                }
            },
            "spec": {
                "ports": [
                    {
                        "protocol": "TCP",
                        "port": 80,
                        "targetPort": 80
                    }
                ],
                "portalIP": "10.0.0.1",
                "sessionAffinity": "None"
            },
            "status": {}
        },
        {
            "kind": "Service",
            "apiVersion": "v1",
            "metadata": {
                "name": "redis-master",
                "namespace": "default",
                "selfLink": "/api/v1/namespaces/default/services/redis-master",
                "uid": "a6fde246-ff78-11e4-8f2d-080027213276",
                "resourceVersion": "72",
                "creationTimestamp": "2015-05-21T05:17:19Z",
                "labels": {
                    "name": "redis-master"
                }
            },
            "spec": {
                "ports": [
                    {
                        "protocol": "TCP",
                        "port": 6379,
                        "targetPort": 6379
                    }
                ],
                "selector": {
                    "name": "redis-master"
                },
                "portalIP": "10.0.0.124",
                "sessionAffinity": "None"
            },
            "status": {}
        }
    ]
};

    // Uses promises
    return {
      loadAll: function() {
        // Simulate async call
        return $q.when(services);
      }
    };
  }

})();
