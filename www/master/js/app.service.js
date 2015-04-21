app.service('SidebarService', [
  '$rootScope',
  function($rootScope) {
    var service = this;
    service.sidebarItems = [];

    service.clearSidebarItems = function() { service.sidebarItems = []; };

    service.renderSidebar = function() {
      var _entries = '';
      service.sidebarItems.forEach(function(entry) { _entries += entry.Html; });

      if (_entries) {
        $rootScope.sidenavLeft = '<div layout="column">' + _entries + '</div>';
      }
    };

    service.addSidebarItem = function(item) {

      service.sidebarItems.push(item);

      service.sidebarItems.sort(function(a, b) { return (a.order > b.order) ? 1 : ((b.order > a.order) ? -1 : 0); });
    };
  }
]);
