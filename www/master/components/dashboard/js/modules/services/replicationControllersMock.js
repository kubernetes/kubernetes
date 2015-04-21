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
      "kind": "ReplicationControllerList",
      "creationTimestamp": null,
      "selfLink": "/api/v1beta1/replicationControllers",
      "resourceVersion": 166552,
      "apiVersion": "v1beta1",
      "items": []
    };

    // Uses promises
    return {
      loadAll: function() {
        // Simulate async call
        return $q.when(replicationControllers);
      }
    };
  }

})();
