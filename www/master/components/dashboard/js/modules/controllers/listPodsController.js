

app.controller('ListPodsCtrl', [
  '$scope',
  '$routeParams',
  'k8sApi',
  'lodash',
  '$location',
  function($scope, $routeParams, k8sApi, lodash, $location) {
    var _ = lodash;
    $scope.getData = getData;
    $scope.loading = true;
    $scope.k8sApi = k8sApi;
    $scope.pods = null;
    $scope.groupedPods = null;
    $scope.serverView = false;

    $scope.headers = [
      {name: 'Pod', field: 'pod'},
      {name: 'IP', field: 'ip'},
      {name: 'Status', field: 'status'},
      {name: 'Containers', field: 'containers'},
      {name: 'Images', field: 'images'},
      {name: 'Host', field: 'host'},
      {name: 'Labels', field: 'labels'}
    ];

    $scope.custom = {
      pod: '',
      ip: 'grey',
      containers: 'grey',
      images: 'grey',
      host: 'grey',
      labels: 'grey',
      status: 'grey'
    };
    $scope.sortable = ['pod', 'ip', 'status'];
    $scope.count = 10;

    $scope.go = function(data) { $location.path('/dashboard/pods/' + data.pod); };

    var orderedPodNames = [];

    function handleError(data, status, headers, config) {
      console.log("Error (" + status + "): " + data);
      $scope.loading = false;
    };

    function getPodName(pod) { return _.has(pod.metadata.labels, 'name') ? pod.metadata.labels.name : pod.metadata.name; }

    $scope.content = [];

    function getData() {
      $scope.loading = true;
      k8sApi.getPods().success(angular.bind(this, function(data) {
        $scope.loading = false;

        var _fixComma = function(str) {
          if (str.substring(0, 1) == ',') {
            return str.substring(1);
          } else {
            return str;
          }
        };

        data.items.forEach(function(pod) {
          var _containers = '', _images = '', _labels = '', _uses = '';

          if (pod.spec) {
            Object.keys(pod.spec.containers)
                .forEach(function(key) {
                  _containers += ', ' + pod.spec.containers[key].name;
                  _images += ', ' + pod.spec.containers[key].image;
                });
          }

          if (pod.metadata.labels) {
            Object.keys(pod.metadata.labels)
                .forEach(function(key) {
                  if (key == 'name') {
                    _labels += ', ' + pod.metadata.labels[key];
                  }
                  if (key == 'uses') {
                    _uses += ', ' + pod.metadata.labels[key];
                  }
                });
            }

          $scope.content.push({
            pod: pod.metadata.name,
            ip: pod.status.podIP,
            containers: _fixComma(_containers),
            images: _fixComma(_images),
            host: pod.spec.host,
            labels: _fixComma(_labels) + ':' + _fixComma(_uses),
            status: pod.status.phase
          });

        });

      })).error(angular.bind(this, handleError));
    };

    $scope.getPodRestarts = function(pod) {
      var r = null;
      var container = _.first(pod.spec.containers);
      if (container) r = pod.status.containerStatuses[container.name].restartCount;
      return r;
    };

    $scope.otherLabels = function(labels) { return _.omit(labels, 'name') };

    $scope.podStatusClass = function(pod) {

      var s = pod.status.phase.toLowerCase();

      if (s == 'running' || s == 'succeeded')
        return null;
      else
        return "status-" + s;
    };

    $scope.podIndexFromName = function(pod) {
      var name = getPodName(pod);
      return _.indexOf(orderedPodNames, name) + 1;
    };

    getData();

  }
]);
