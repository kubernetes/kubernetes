/**=========================================================
 * Module: tabs-global.js
 * Page Controller
 =========================================================*/

app.controller('TabCtrl', [
  '$scope',
  '$location',
  'tabs',
  function($scope, $location, tabs) {
    $scope.tabs = tabs;

    $scope.switchTab = function(index) {
      var location_path = $location.path();
      var tab = tabs[index];

      if (tab) {
        var path = '/%s'.format(tab.component);
        if (location_path.indexOf(path) == -1) {
          $location.path(path);
        }
      }
    };
  }
]);
