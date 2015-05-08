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
      "kind": "ServiceList",
      "creationTimestamp": null,
      "selfLink": "/api/v1beta1/services",
      "resourceVersion": 166552,
      "apiVersion": "v1beta1",
      "items": [
        {
          "id": "kubernetes",
          "uid": "626dd08d-ab51-11e4-8ae8-061695c59fcf",
          "creationTimestamp": "2015-02-03T03:04:36Z",
          "selfLink": "/api/v1beta1/services/kubernetes?namespace=default",
          "resourceVersion": 11,
          "namespace": "default",
          "port": 443,
          "protocol": "TCP",
          "labels": {"component": "apiserver", "provider": "kubernetes"},
          "selector": null,
          "containerPort": 0,
          "portalIP": "10.244.66.215",
          "sessionAffinity": "None"
        },
        {
          "id": "kubernetes-ro",
          "uid": "626f9584-ab51-11e4-8ae8-061695c59fcf",
          "creationTimestamp": "2015-02-03T03:04:36Z",
          "selfLink": "/api/v1beta1/services/kubernetes-ro?namespace=default",
          "resourceVersion": 12,
          "namespace": "default",
          "port": 80,
          "protocol": "TCP",
          "labels": {"component": "apiserver", "provider": "kubernetes"},
          "selector": null,
          "containerPort": 0,
          "portalIP": "10.244.182.142",
          "sessionAffinity": "None"
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
