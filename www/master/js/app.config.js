angular.module('kubernetesApp.config', []);
angular.module('kubernetesApp.services', ['kubernetesApp.config']);

app.config([
  '$routeProvider',
  function($routeProvider) {
    $routeProvider.when("/404", {templateUrl: "views/partials/404.html"})
        // else 404
        .otherwise({redirectTo: "/404"});
  }
])
    .config([
      '$routeProvider',
      'manifestRoutes',
      function($routeProvider, manifestRoutes) {
        angular.forEach(manifestRoutes, function(r) {
          var route = {
            templateUrl: r.templateUrl
          };
          if (r.controller) {
            route.controller = r.controller;
          }
          if (r.css) {
            route.css = r.css;
          }
          $routeProvider.when(r.url, route);
        });
      }
    ]);

app.value("sections", [{"name":"Dashboard","url":"/dashboard","type":"link","templateUrl":"/components/dashboard/pages/home.html"},{"name":"Dashboard","type":"heading","children":[{"name":"Dashboard","type":"toggle","url":"/dashboard","templateUrl":"/components/dashboard/pages/home.html","pages":[{"name":"Pods","url":"/dashboard/pods","templateUrl":"/components/dashboard/views/listPods.html","type":"link"},{"name":"Pod Visualizer","url":"/dashboard/visualpods","templateUrl":"/components/dashboard/views/listPodsVisualizer.html","type":"link"},{"name":"Services","url":"/dashboard/services","templateUrl":"/components/dashboard/views/listServices.html","type":"link"},{"name":"Replication Controllers","url":"/dashboard/replicationcontrollers","templateUrl":"/components/dashboard/views/listReplicationControllers.html","type":"link"},{"name":"Events","url":"/dashboard/events","templateUrl":"/components/dashboard/views/listEvents.html","type":"link"},{"name":"Nodes","url":"/dashboard/nodes","templateUrl":"/components/dashboard/views/listMinions.html","type":"link"},{"name":"Replication Controller","url":"/dashboard/replicationcontrollers/:replicationControllerId","templateUrl":"/components/dashboard/views/replication.html","type":"link"},{"name":"Service","url":"/dashboard/services/:serviceId","templateUrl":"/components/dashboard/views/service.html","type":"link"},{"name": "Node","url": "/dashboard/nodes/:nodeId","templateUrl": "/components/dashboard/views/node.html","type": "link"},{"name":"Explore","url":"/dashboard/groups/:grouping*?/selector/:selector*?","templateUrl":"/components/dashboard/views/groups.html","type":"link"},{"name":"Pod","url":"/dashboard/pods/:podId","templateUrl":"/components/dashboard/views/pod.html","type":"link"}]}]},{"name":"Graph","url":"/graph","type":"link","templateUrl":"/components/graph/pages/home.html"},{"name":"Graph","url":"/graph/inspect","type":"link","templateUrl":"/components/graph/pages/inspect.html","css":"/components/graph/css/show-details-table.css"},{"name":"Graph","type":"heading","children":[{"name":"Graph","type":"toggle","url":"/graph","templateUrl":"/components/graph/pages/home.html","pages":[{"name":"Test","url":"/graph/test","type":"link","templateUrl":"/components/graph/pages/home.html"}]}]}]);
