app.directive('includeReplace',
              function() {
                'use strict';
                return {
                  require: 'ngInclude',
                  restrict: 'A', /* optional */
                  link: function(scope, el, attrs) { el.replaceWith(el.children()); }
                };
              })
    .directive('compile',
               function($compile) {
                 'use strict';
                 return function(scope, element, attrs) {
                   scope.$watch(function(scope) { return scope.$eval(attrs.compile); },
                                function(value) {
                                  element.html(value);
                                  $compile(element.contents())(scope);
                                });
                 };
               })
    .directive("kubernetesUiMenu",
               function() {
                 'use strict';
                 return {
                   templateUrl: "views/partials/kubernetes-ui-menu.tmpl.html"
                 };
               })
    .directive('menuToggle', function() {
      'use strict';
      return {
        scope: {section: '='},
        templateUrl: 'views/partials/menu-toggle.tmpl.html',
        link: function($scope, $element) {
          var controller = $element.parent().controller();

          $scope.isOpen = function() { return controller.isOpen($scope.section); };
          $scope.toggle = function() { controller.toggleOpen($scope.section); };

          var parentNode = $element[0].parentNode.parentNode.parentNode;
          if (parentNode.classList.contains('parent-list-item')) {
            var heading = parentNode.querySelector('h2');
            $element[0].firstChild.setAttribute('aria-describedby', heading.id);
          }
        }
      };
    });

app.filter('startFrom',
           function() {
             'use strict';
             return function(input, start) { return input.slice(start); };
           })
    .filter('nospace', function() {
      'use strict';
      return function(value) { return (!value) ? '' : value.replace(/ /g, ''); };
    });
