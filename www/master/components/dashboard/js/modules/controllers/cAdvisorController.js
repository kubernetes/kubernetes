

app.controller('cAdvisorController', [
  '$scope',
  '$routeParams',
  'k8sApi',
  'lodash',
  'cAdvisorService',
  '$q',
  '$interval',
  function($scope, $routeParams, k8sApi, lodash, cAdvisorService, $q, $interval) {
    $scope.k8sApi = k8sApi;

    $scope.activeMinionDataById = {};
    $scope.maxDataByById = {};

    $scope.getData = function() {
      $scope.loading = true;

      k8sApi.getMinions().success(angular.bind(this, function(res) {
        $scope.minions = res;
        // console.log(res);
        var promises = lodash.map(res.items, function(m) { return cAdvisorService.getDataForMinion(m.id); });

        $q.all(promises).then(
            function(dataArray) {
              lodash.each(dataArray, function(data, i) {
                var m = res.items[i];

                var maxData = maxMemCpuInfo(m.id, data.memoryData, data.cpuData, data.filesystemData);

                // console.log("maxData", maxData);

                $scope.activeMinionDataById[m.id] =
                    transformMemCpuInfo(data.memoryData, data.cpuData, data.filesystemData, maxData, m.hostIP)
              });

            },
            function(errorData) {
              // console.log("Error: " + errorData);
              $scope.loading = false;
            });

        $scope.loading = false;
      })).error(angular.bind(this, this.handleError));
    };

    function getcAdvisorDataForMinion(m) {
      var p = cAdvisorService.getDataForMinion(m.hostIP);
      return p;
    }

    function handleError(data, status, headers, config) {
      // console.log("Error (" + status + "): " + data);
      $scope.loading = false;
    };

    // d3
    function getColorForIndex(i, percentage) {
      // var colors = ['red', 'blue', 'yellow', 'pink', 'purple', 'green', 'orange'];
      // return colors[i];
      var c = "color-" + (i + 1);
      if (percentage && percentage >= 90)
        c = c + ' color-critical';
      else if (percentage && percentage >= 80)
        c = c + ' color-warning';

      return c;
    }

    function getMaxColorForIndex(i, percentage) {
      // var colors = ['red', 'blue', 'yellow', 'pink', 'purple', 'green', 'orange'];
      // return colors[i];
      var c = "color-max-" + (i + 1);
      if (percentage && percentage >= 90)
        c = c + ' color-max-critical';
      else if (percentage && percentage >= 80)
        c = c + ' color-max-warning';

      return c;
    }

    function maxMemCpuInfo(mId, mem, cpu, fsDataArray) {
      if ($scope.maxDataByById[mId] === undefined) $scope.maxDataByById[mId] = {};

      var currentMem = mem.current;
      var currentCpu = cpu;

      var items = [];

      if ($scope.maxDataByById[mId]['cpu'] === undefined ||
          $scope.maxDataByById[mId]['cpu'] < currentCpu.cpuPercentUsage) {
        // console.log("New max cpu " + mId, $scope.maxDataByById[mId].cpu, currentCpu.cpuPercentUsage);
        $scope.maxDataByById[mId]['cpu'] = currentCpu.cpuPercentUsage;
      }
      items.push({
        maxValue: $scope.maxDataByById[mId]['cpu'],
        maxTickClassNames: getColorForIndex(0, $scope.maxDataByById[mId]['cpu']),
        maxClassNames: getMaxColorForIndex(0, $scope.maxDataByById[mId]['cpu'])
      });

      var memPercentage = Math.floor((currentMem.memoryUsage * 100.0) / currentMem.memoryLimit);
      if ($scope.maxDataByById[mId]['mem'] === undefined || $scope.maxDataByById[mId]['mem'] < memPercentage)
        $scope.maxDataByById[mId]['mem'] = memPercentage;
      items.push({
        maxValue: $scope.maxDataByById[mId]['mem'],
        maxTickClassNames: getColorForIndex(1, $scope.maxDataByById[mId]['mem']),
        maxClassNames: getMaxColorForIndex(1, $scope.maxDataByById[mId]['mem'])
      });

      for (var i = 0; i < fsDataArray.length; i++) {
        var f = fsDataArray[i];
        var fid = 'FS #' + f.filesystemNumber;
        if ($scope.maxDataByById[mId][fid] === undefined || $scope.maxDataByById[mId][fid] < f.totalUsage)
          $scope.maxDataByById[mId][fid] = f.totalUsage;
        items.push({
          maxValue: $scope.maxDataByById[mId][fid],
          maxTickClassNames: getColorForIndex(2 + i, $scope.maxDataByById[mId][fid]),
          maxClassNames: getMaxColorForIndex(2 + i, $scope.maxDataByById[mId][fid])
        });
      }

      // console.log("Max Data is now " + mId, $scope.maxDataByById[mId]);
      return items;
    }

    function transformMemCpuInfo(mem, cpu, fsDataArray, maxData, hostName) {
      var currentMem = mem.current;
      var currentCpu = cpu;

      var items = [];

      items.push({
        label: 'CPU',
        stats: currentCpu.cpuPercentUsage + '%',
        value: currentCpu.cpuPercentUsage,
        classNames: getColorForIndex(0, currentCpu.cpuPercentUsage),
        maxData: maxData[0],
        hostName: hostName
      });

      var memPercentage = Math.floor((currentMem.memoryUsage * 100.0) / currentMem.memoryLimit);
      items.push({
        label: 'Memory',
        stats: currentMem.memoryUsageDescription + ' / ' + currentMem.memoryLimitDescription,
        value: memPercentage,
        classNames: getColorForIndex(1, memPercentage),
        maxData: maxData[1],
        hostName: hostName
      });

      for (var i = 0; i < fsDataArray.length; i++) {
        var f = fsDataArray[i];

        items.push({
          label: 'FS #' + f.filesystemNumber,
          stats: f.usageDescription + ' / ' + f.capacityDescription,
          value: f.totalUsage,
          classNames: getColorForIndex(2 + i, f.totalUsage),
          maxData: maxData[2 + i],
          hostName: hostName

        });
      }

      var a = [];
      var segments = {
        segments: items
      };
      a.push(segments);
      return a;
    };

    // end d3
    var promise = $interval($scope.getData, 3000);

    // Cancel interval on page changes
    $scope.$on('$destroy', function() {
      if (angular.isDefined(promise)) {
        $interval.cancel(promise);
        promise = undefined;
      }
    });

    $scope.getData();

  }
]);
