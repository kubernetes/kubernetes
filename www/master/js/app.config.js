angular.module('kubernetesApp.config', []);
angular.module('kubernetesApp.services', ['kubernetesApp.config']);

app.config([
  '$routeProvider',
  function($routeProvider) {
    $routeProvider.when("/404", {templateUrl: "/views/partials/404.html"})
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
