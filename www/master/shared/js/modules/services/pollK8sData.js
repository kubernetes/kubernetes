(function() {
  "use strict";

  var pollK8sDataServiceProvider = function PollK8sDataServiceProvider(_) {
    // A set of configuration controlling the polling behavior.
    // Their values should be configured in the application before
    // creating the service instance.

    var useSampleData = false;
    this.setUseSampleData = function(value) { useSampleData = value; };

    var sampleDataFiles = [
      "assets/data/sampleData1.json"
    ];
    this.setSampleDataFiles = function(value) { sampleDataFiles = value; };

    var proxyVerb = "/api/v1/proxy/namespaces/default/services";

    var dataService = undefined;
    var dataServiceDefault = "/cluster-insight";

    this.setDataService = function(value) { dataService = value; };
    this.getDataService = function() { return dataService || dataServiceDefault; }
    var getDataService = this.getDataService;

    var dataServicePortName = undefined;
    var dataServicePortNameDefault = ":cluster-insight";

    this.setDataServicePortName = function(value) { dataServicePortName = value; };
    this.getDataServicePortName = function() { return dataServicePortName || dataServicePortNameDefault; }
    var getDataServicePortName = this.getDataServicePortName;

    var dataServiceEndpoint = undefined;
    var dataServiceEndpointDefault = "/cluster";

    this.setDataServiceEndpoint = function(value) { dataServiceEndpoint = value; };
    this.getDataServiceEndpoint = function() { return dataServiceEndpoint || dataServiceEndpointDefault; }
    var getDataServiceEndpoint = this.getDataServiceEndpoint;

    var pollMinIntervalSec = undefined;
    var pollMinIntervalSecDefault = 10;

    this.setPollMinIntervalSec = function(value) { pollMinIntervalSec = value; };
    this.getPollMinIntervalSec = function() { return pollMinIntervalSec || pollMinIntervalSecDefault; };
    var getPollMinIntervalSec = this.getPollMinIntervalSec;

    var pollMaxIntervalSec = undefined;
    var pollMaxIntervalSecDefault = 120;

    this.setPollMaxIntervalSec = function(value) { pollMaxIntervalSec = value; };
    this.getPollMaxIntervalSec = function() { return pollMaxIntervalSec || pollMaxIntervalSecDefault; };
    var getPollMaxIntervalSec = this.getPollMaxIntervalSec;

    var pollErrorThreshold = undefined;
    var pollErrorThresholdDefault = 5;

    this.setPollErrorThreshold = function(value) { pollErrorThreshold = value; };
    this.getPollErrorThreshold = function() { return pollErrorThreshold || pollErrorThresholdDefault; };
    var getPollErrorThreshold = this.getPollErrorThreshold;

    this.$get = function($http, $timeout) {
      // Now the sequenceNumber will be used for debugging and verification purposes.
      var k8sdatamodel = {
        "data": undefined,
        "sequenceNumber": 0,
        "useSampleData": useSampleData
      };
      var pollingError = 0;
      var promise = undefined;

      // Implement fibonacci back off when the service is down.
      var pollInterval = getPollMinIntervalSec();
      var pollIncrement = pollInterval;

      // Reset polling interval.
      var resetCounters = function() {
        pollInterval = getPollMinIntervalSec();
        pollIncrement = pollInterval;
      };

      // Bump error count and polling interval.
      var bumpCounters = function() {
        // Bump the error count.
        pollingError++;

        // TODO: maybe display an error in the UI to the end user.
        if (pollingError % getPollErrorThreshold() === 0) {
          console.log("Error: " + pollingError + " consecutive polling errors for " + getDataService() + ".");
        }

        // Bump the polling interval.
        var oldIncrement = pollIncrement;
        pollIncrement = pollInterval;
        pollInterval += oldIncrement;

        // Reset when limit reached.
        if (pollInterval > getPollMaxIntervalSec()) {
          resetCounters();
        }
      };

      var updateModel = function(newModel) {
        var dedupe = function(dataModel) {
          if (dataModel.resources) {
            var compareResources = function(resource) { return resource.id; };
            dataModel.resources = _.chain(dataModel.resources)
              .sortBy(compareResources)
              .uniq(true, compareResources)
              .value();
          }

          if (dataModel.relations) {
            var compareRelations = function(relation) { return relation.source + relation.target; };
            dataModel.relations = _.chain(dataModel.relations)
              .sortBy(compareRelations)
              .uniq(true, compareRelations)
              .value();
          }
        };

        dedupe(newModel);

        var newModelString = JSON.stringify(newModel);
        var oldModelString = "";
        if (k8sdatamodel.data) {
          oldModelString = JSON.stringify(k8sdatamodel.data);
        }

        if (newModelString !== oldModelString) {
          k8sdatamodel.data = newModel;
          k8sdatamodel.sequenceNumber++;
        }

        pollingError = 0;
        resetCounters();
      };

      var nextSampleDataFile = 0;
      var getSampleDataFile = function() {
        var result = "";
        if (sampleDataFiles.length > 0) {
          result = sampleDataFiles[nextSampleDataFile % sampleDataFiles.length];
          ++nextSampleDataFile;
        }

        return result;
      };

      var getDataServiceURL = function() {
        return proxyVerb + getDataService() + getDataServicePortName() + getDataServiceEndpoint();
      };

      var pollOnce = function(scope, repeat) {
        var dataSource = (k8sdatamodel.useSampleData) ? getSampleDataFile() : getDataServiceURL();
        if (dataSource) {
          $.getJSON(dataSource)
              .done(function(newModel, jqxhr, textStatus) {
                if (newModel && newModel.success) {
                  delete newModel.success;
                  delete newModel.timestamp;  // Remove changing timestamp.
                  updateModel(newModel);
                  scope.$apply();
                  promise = repeat ? $timeout(function() { pollOnce(scope, true); }, pollInterval * 1000) : undefined;
                  return;
                }

                bumpCounters();
                promise = repeat ? $timeout(function() { pollOnce(scope, true); }, pollInterval * 1000) : undefined;
              })
              .fail(function(jqxhr, textStatus, error) {
                bumpCounters();
                promise = repeat ? $timeout(function() { pollOnce(scope, true); }, pollInterval * 1000) : undefined;
              });
        }
      };

      var isPolling = function() { return promise ? true : false; };

      var start = function(scope) {
        // If polling has already started, then calling start() again would
        // just reset the counters and polling interval, but it will not
        // start a new thread polling in parallel to the existing polling
        // thread.
        resetCounters();
        if (!promise) {
          k8sdatamodel.data = undefined;
          pollOnce(scope, true);
        }
      };

      var stop = function() {
        if (promise) {
          $timeout.cancel(promise);
          promise = undefined;
        }
      };

      var refresh = function(scope) {
        stop();
        resetCounters();
        k8sdatamodel.data = undefined;
        pollOnce(scope, false);
      };

      return {
        "k8sdatamodel": k8sdatamodel,
        "isPolling": isPolling,
        "refresh": refresh,
        "start": start,
        "stop": stop
      };
    };
  };

  angular.module("kubernetesApp.services")
      .provider("pollK8sDataService", ["lodash", pollK8sDataServiceProvider])
      .config(function(pollK8sDataServiceProvider, ENV) {
        if (ENV && ENV['/']) {
          if (ENV['/']['k8sDataService']) {
            pollK8sDataServiceProvider.setDataService(ENV['/']['k8sDataService']);
          }
          if (ENV['/']['k8sDataServicePortName']) {
            pollK8sDataServiceProvider.setDataServicePortName(ENV['/']['k8sDataServicePortName']);
          }
          if (ENV['/']['k8sDataServiceEndpoint']) {
            pollK8sDataServiceProvider.setDataServiceEndpoint(ENV['/']['k8sDataServiceEndpoint']);
          }
          if (ENV['/']['k8sDataPollIntervalMinSec']) {
            pollK8sDataServiceProvider.setPollIntervalSec(ENV['/']['k8sDataPollIntervalMinSec']);
          }
          if (ENV['/']['k8sDataPollIntervalMaxSec']) {
            pollK8sDataServiceProvider.setPollIntervalSec(ENV['/']['k8sDataPollIntervalMaxSec']);
          }
          if (ENV['/']['k8sDataPollErrorThreshold']) {
            pollK8sDataServiceProvider.setPollErrorThreshold(ENV['/']['k8sDataPollErrorThreshold']);
          }
        }
      });

}());
